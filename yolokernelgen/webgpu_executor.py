"""WebGPU kernel execution wrapper."""

import numpy as np
from typing import List, Tuple, Optional

# Try to import pydawn, but make it optional
try:
    from pydawn import utils, webgpu  # type: ignore
    PYDAWN_AVAILABLE = True
except ImportError:
    PYDAWN_AVAILABLE = False
    utils = None  # type: ignore
    webgpu = None  # type: ignore


def tensor_to_bytes(tensor: np.ndarray) -> bytes:
    """Convert numpy tensor to bytes."""
    return tensor.astype(np.float32).tobytes()


def bytes_to_tensor(buffer_bytes: bytes, shape: List[int], dtype: str = "float32") -> np.ndarray:
    """Convert bytes buffer back to numpy tensor."""
    return np.frombuffer(buffer_bytes, dtype=dtype).reshape(shape)


def prepare_buffers(tensors: List[np.ndarray]) -> List[bytes]:
    """Prepare multiple tensors as byte buffers."""
    return [tensor_to_bytes(t) for t in tensors]


def calculate_workgroups(
    num_elements: int,
    workgroup_size: int = 256,
    max_per_dim: int = 65535
) -> Tuple[int, int, int]:
    """Calculate workgroup dispatch dimensions."""
    total_workgroups = (num_elements + workgroup_size - 1) // workgroup_size

    if total_workgroups <= max_per_dim:
        return total_workgroups, 1, 1
    else:
        workgroups_x = max_per_dim
        workgroups_y = (total_workgroups + max_per_dim - 1) // max_per_dim
        return workgroups_x, workgroups_y, 1


def execute_kernel(
    kernel_source: str,
    input_tensors: List[np.ndarray],
    output_shape: Optional[List[int]] = None,
    workgroup_size: int = 256
) -> np.ndarray:
    """Execute WebGPU kernel with multiple input tensors."""

    # Create adapter and device
    adapter = utils.request_adapter_sync(  # type: ignore
        power_preference=webgpu.WGPUPowerPreference_HighPerformance  # type: ignore
    )
    dev = utils.request_device_sync(adapter, [])  # type: ignore

    # Prepare input buffers
    input_buffers_bytes = prepare_buffers(input_tensors)

    # Calculate output size
    if output_shape:
        output_elements = int(np.prod(output_shape))
    else:
        # Assume output same size as first input
        output_elements = input_tensors[0].size
        output_shape = list(input_tensors[0].shape)

    output_size = output_elements * 4  # 4 bytes per float32

    # Align all buffer sizes to 16-byte boundary
    aligned_sizes = []
    aligned_buffers = []

    for buf_bytes in input_buffers_bytes:
        original_size = len(buf_bytes)
        aligned_size = ((original_size + 15) // 16) * 16
        if aligned_size > original_size:
            aligned_buf = buf_bytes + b'\x00' * (aligned_size - original_size)
        else:
            aligned_buf = buf_bytes
        aligned_sizes.append(aligned_size)
        aligned_buffers.append(aligned_buf)

    # Align output size
    aligned_output_size = ((output_size + 15) // 16) * 16

    # Create shader module
    shader_module = utils.create_shader_module(dev, kernel_source)  # type: ignore

    # Create GPU buffers
    gpu_input_buffers = []
    for size in aligned_sizes:
        gpu_input_buffers.append(utils.create_buffer(  # type: ignore
            dev,
            size,
            webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst  # type: ignore
        ))

    gpu_output_buffer = utils.create_buffer(  # type: ignore
        dev,
        aligned_output_size,
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc  # type: ignore
    )

    # Write data to GPU buffers
    for gpu_buf, aligned_buf in zip(gpu_input_buffers, aligned_buffers):
        utils.write_buffer(dev, gpu_buf, 0, bytearray(aligned_buf))  # type: ignore

    # Setup bind group layout entries
    binding_layouts = []
    for i in range(len(input_tensors)):
        binding_layouts.append({
            "binding": i,
            "visibility": webgpu.WGPUShaderStage_Compute,  # type: ignore  # type: ignore
            "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},  # type: ignore
        })

    # Add output buffer binding
    binding_layouts.append({
        "binding": len(input_tensors),
        "visibility": webgpu.WGPUShaderStage_Compute,  # type: ignore
        "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},  # type: ignore
    })

    # Setup bindings
    bindings = []
    for i, (gpu_buf, size) in enumerate(zip(gpu_input_buffers, aligned_sizes)):
        bindings.append({
            "binding": i,
            "resource": {"buffer": gpu_buf, "offset": 0, "size": size},
        })

    # Add output binding
    bindings.append({
        "binding": len(input_tensors),
        "resource": {"buffer": gpu_output_buffer, "offset": 0, "size": aligned_output_size},
    })

    # Create bind group and pipeline
    bind_group_layout = utils.create_bind_group_layout(device=dev, entries=binding_layouts)  # type: ignore
    pipeline_layout = utils.create_pipeline_layout(device=dev, bind_group_layouts=[bind_group_layout])  # type: ignore
    bind_group = utils.create_bind_group(device=dev, layout=bind_group_layout, entries=bindings)  # type: ignore

    # Create compute pipeline
    compute_pipeline = utils.create_compute_pipeline(  # type: ignore
        device=dev,
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )

    # Create and run command encoder
    command_encoder = utils.create_command_encoder(dev)  # type: ignore
    compute_pass = utils.begin_compute_pass(command_encoder)  # type: ignore
    utils.set_pipeline(compute_pass, compute_pipeline)  # type: ignore
    utils.set_bind_group(compute_pass, bind_group)  # type: ignore

    # Calculate workgroup dispatch
    workgroups_x, workgroups_y, workgroups_z = calculate_workgroups(
        output_elements, workgroup_size
    )

    utils.dispatch_workgroups(compute_pass, workgroups_x, workgroups_y, workgroups_z)  # type: ignore
    utils.end_compute_pass(compute_pass)  # type: ignore
    cb_buffer = utils.command_encoder_finish(command_encoder)  # type: ignore

    # Submit and wait
    utils.submit(dev, [cb_buffer])  # type: ignore
    utils.sync(dev)  # type: ignore

    # Read result and convert back to tensor
    result_buffer = utils.read_buffer(dev, gpu_output_buffer)  # type: ignore
    return bytes_to_tensor(bytes(result_buffer[:output_size]), output_shape)


def execute_simple_kernel(kernel_source: str, input_bytes: bytes, num_elements: int) -> bytes:
    """Simple kernel execution for single input/output (compatibility function)."""
    # Create adapter and device
    adapter = utils.request_adapter_sync(  # type: ignore
        power_preference=webgpu.WGPUPowerPreference_HighPerformance  # type: ignore
    )
    dev = utils.request_device_sync(adapter, [])  # type: ignore

    original_size = len(input_bytes)

    # Align buffer size
    aligned_size = ((original_size + 15) // 16) * 16
    if aligned_size > original_size:
        padded_bytes = input_bytes + b'\x00' * (aligned_size - original_size)
    else:
        padded_bytes = input_bytes

    # Create shader module
    shader_module = utils.create_shader_module(dev, kernel_source)  # type: ignore

    # Create buffers
    input_buffer = utils.create_buffer(  # type: ignore
        dev,
        aligned_size,
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst  # type: ignore
    )
    output_buffer = utils.create_buffer(  # type: ignore
        dev,
        aligned_size,
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc  # type: ignore
    )

    # Write data
    utils.write_buffer(dev, input_buffer, 0, bytearray(padded_bytes))  # type: ignore

    # Setup bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": webgpu.WGPUShaderStage_Compute,  # type: ignore  # type: ignore
            "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},  # type: ignore
        },
        {
            "binding": 1,
            "visibility": webgpu.WGPUShaderStage_Compute,  # type: ignore  # type: ignore
            "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},  # type: ignore
        },
    ]

    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": input_buffer, "offset": 0, "size": aligned_size},
        },
        {
            "binding": 1,
            "resource": {"buffer": output_buffer, "offset": 0, "size": aligned_size},
        },
    ]

    # Create pipeline
    bind_group_layout = utils.create_bind_group_layout(device=dev, entries=binding_layouts)  # type: ignore
    pipeline_layout = utils.create_pipeline_layout(device=dev, bind_group_layouts=[bind_group_layout])  # type: ignore
    bind_group = utils.create_bind_group(device=dev, layout=bind_group_layout, entries=bindings)  # type: ignore

    compute_pipeline = utils.create_compute_pipeline(  # type: ignore
        device=dev,
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )

    # Execute
    command_encoder = utils.create_command_encoder(dev)  # type: ignore
    compute_pass = utils.begin_compute_pass(command_encoder)  # type: ignore
    utils.set_pipeline(compute_pass, compute_pipeline)  # type: ignore
    utils.set_bind_group(compute_pass, bind_group)  # type: ignore

    workgroups_x, workgroups_y, workgroups_z = calculate_workgroups(num_elements)
    utils.dispatch_workgroups(compute_pass, workgroups_x, workgroups_y, workgroups_z)  # type: ignore

    utils.end_compute_pass(compute_pass)  # type: ignore
    cb_buffer = utils.command_encoder_finish(command_encoder)  # type: ignore

    utils.submit(dev, [cb_buffer])  # type: ignore
    utils.sync(dev)  # type: ignore

    # Read result
    result_buffer = utils.read_buffer(dev, output_buffer)  # type: ignore
    return bytes(result_buffer[:original_size])