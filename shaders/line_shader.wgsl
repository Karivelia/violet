struct VertexInput {
    @location(0) pos_2d: vec2<f32>,
    @location(1) z_index: f32,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>
}

@vertex
fn vertex_main(input: VertexInput)->VertexOutput{
    var output: VertexOutput;
    output.clip_position = vec4<f32>(input.pos_2d, input.z_index, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fragment_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(vec3<f32>(color.rgb) / 255.0, 1.0);
}