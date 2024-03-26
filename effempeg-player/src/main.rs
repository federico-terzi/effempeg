use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SampleFormat, SampleRate,
};
use std::{
    borrow::Cow,
    collections::VecDeque,
    fs::File,
    io::BufWriter,
    num::NonZeroU32,
    sync::{mpsc::TrySendError, Arc, Mutex},
    time::Instant,
};
use wgpu::util::DeviceExt;
use winit::{
    dpi::Size,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use effempeg::{
    format::{input, Pixel},
    frame::Audio,
    media::Type,
    rescale,
    software::scaling::{context::Context, flag::Flags},
    util::frame::video::Video,
    ChannelLayout, Rescale,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Scene {
    width: f32,
    height: f32,
    _padding: [f32; 2],
    vertexes: [Vertex; 4],
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut VERTICES: Vec<Vertex> = vec![
        Vertex {
            position: [-0.6, -0.6, 0.0, 0.0],
        },
        Vertex {
            position: [0.6, -0.6, 0.0, 0.0],
        },
        Vertex {
            position: [0.6, 0.6, 0.0, 0.0],
        },
        Vertex {
            position: [-0.6, 0.6, 0.0, 0.0],
        },
    ];

    const INDICES: &[u16] = &[
        0, 1, 2, // first triangle
        0, 2, 3, // second triangle
    ];

    effempeg::init().unwrap();

    let (video_tx, video_receiver) = std::sync::mpsc::sync_channel(5);
    let (audio_tx, audio_receiver) = std::sync::mpsc::sync_channel(50);

    let mut ictx = input(std::env::args().nth(1).expect("Cannot open file.")).unwrap();

    let video_input = ictx
        .streams()
        .best(Type::Video)
        .ok_or(effempeg::Error::StreamNotFound)
        .unwrap();
    let video_stream_index = video_input.index();

    let audio_input = ictx
        .streams()
        .best(Type::Audio)
        .ok_or(effempeg::Error::StreamNotFound)
        .unwrap();
    let audio_stream_index = audio_input.index();

    let mut video_decoder = video_input.decoder().unwrap().video().unwrap();
    let video_timebase = video_decoder.time_base().unwrap();
    println!("video timebase {:?}", video_decoder.time_base());
    let mut audio_decoder = audio_input.decoder().unwrap().audio().unwrap();
    let audio_decoder_sample_rate = audio_decoder.sample_rate();
    let audio_timebase = audio_decoder.time_base().unwrap();

    println!("audio format: {:?}", audio_decoder.format());
    println!("audio channels: {:?}", audio_decoder.channel_layout());

    let audio_host = cpal::default_host();
    let audio_device = audio_host
        .default_output_device()
        .expect("no output device available");
    println!("Default output device: {}", audio_device.name().unwrap());

    let mut audio_config = None;
    for format in audio_device.supported_output_configs().unwrap() {
        println!("{:?}", format);
        if format.sample_format() == SampleFormat::F32 {
            audio_config = Some(format);
            break;
        }
    }
    println!("decoder sample rate: {}", audio_decoder.sample_rate());
    println!(
        "sample size: {}",
        audio_config.unwrap().sample_format().sample_size()
    );
    let audio_config = audio_config.unwrap().with_max_sample_rate();
    // .with_sample_rate(SampleRate(audio_decoder.sample_rate()));
    let format = audio_config.sample_format();
    let audio_config = audio_config.config();
    let sample_rate = audio_config.sample_rate.0;
    println!("device sample rate: {}", sample_rate);

    // let mut audio_config = audio_device.default_output_config().unwrap();
    // let mut audio_config = audio_config.config();
    println!("config {:?}", audio_config);
    // for format in audio_device.supported_output_configs().unwrap() {
    //     println!("{:?}", format);
    // }

    let width = video_decoder.width();
    let height = video_decoder.height();
    // let width = 1920;
    // let height = 1080;
    // let width = 640;
    // let height = 480;

    std::thread::spawn(move || {
        let mut scaler = Context::get(
            video_decoder.format(),
            video_decoder.width(),
            video_decoder.height(),
            Pixel::RGBA,
            width,
            height,
            // Flags::POINT,
            Flags::POINT,
        )
        .unwrap();

        let mut resampler = effempeg::software::resampling::context::Context::get(
            audio_decoder.format(),
            audio_decoder.channel_layout(),
            audio_decoder.sample_rate(),
            effempeg::format::Sample::F32(effempeg::format::sample::Type::Packed),
            // effempeg::format::Sample::F32(effempeg::format::sample::Type::Planar),
            // audio_decoder.channel_layout(),
            ChannelLayout::default(2),
            // audio_decoder.sample_rate(),
            sample_rate,
        )
        .unwrap();
        println!(
            "previous sample rate: {} current sample rate: {}",
            audio_decoder.sample_rate(),
            sample_rate
        );
        let ratio = sample_rate as f64 / audio_decoder_sample_rate as f64;

        println!("ratio: {}", ratio);

        // let mut scaler = Context::get(
        //     decoder.format(),
        //     decoder.width(),
        //     decoder.height(),
        //     Pixel::RGB24,
        //     decoder.width(),
        //     decoder.height(),
        //     Flags::BILINEAR,
        // )
        // .unwrap();

        let mut receive_and_process_decoded_video_frames =
            |decoder: &mut effempeg::decoder::Video| -> Result<(), effempeg::Error> {
                let mut decoded = Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    // println!("frame {:?}", decoded.timestamp());
                    // println!("frame2 {:?}", decoded.pts());
                    // video_tx.send(decoded.clone()).unwrap();

                    let mut rgb_frame = Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;
                    rgb_frame.set_pts(decoded.pts());
                    video_tx.send(rgb_frame).unwrap();
                    // match video_tx.try_send(rgb_frame) {
                    //     Ok(_) => {}
                    //     Err(TrySendError::Full(_)) => {
                    //         println!("video channel full, skipping frame!");
                    //     }
                    //     _ => {
                    //         panic!("send error!");
                    //     }
                    // }
                }
                Ok(())
            };

        let mut receive_and_process_decoded_audio_frames =
            |decoder: &mut effempeg::decoder::Audio| -> Result<(), effempeg::Error> {
                let mut decoded = effempeg::frame::Audio::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    // audio_tx.send(decoded.clone()).unwrap();

                    // TODO: investigate why the audio stutters?

                    let mut resampled = effempeg::frame::Audio::empty();
                    // let sample_number = (decoded.samples() as f64 * ratio) as usize;
                    // let mut resampled = effempeg::frame::Audio::new(
                    //     resampler.output().format,
                    //     sample_number,
                    //     resampler.output().channel_layout,
                    // );
                    let result = resampler.run(&decoded, &mut resampled)?;
                    resampled.set_pts(decoded.pts());
                    assert!(resampled.is_packed(), "resampled is not packed");
                    // println!("resampled pts: {:?}", resampled.pts());
                    // println!("resampled planes2: {}", resampled.planes());

                    // println!(
                    //     "result: {:?}, samples: {}, previous size: {} current size {}, current samples: {}",
                    //     result,
                    //     decoded.samples(),
                    //     decoded.data(0).len(),
                    //     resampled.data(0).len(),
                    //     resampled.samples(),
                    // );
                    audio_tx.send(resampled).unwrap();

                    if result.is_some() {
                        loop {
                            let mut resampled = effempeg::frame::Audio::empty();
                            resampled.alloc(
                                resampler.output().format,
                                decoded.samples(),
                                resampler.output().channel_layout,
                            );
                            assert!(!resampled.is_empty(), "resampled is empty");
                            let result = resampler.flush(&mut resampled).unwrap();

                            // TODO: this is very likely incorrect!
                            if let Some(delay) = result {
                                resampled.set_pts(Some(decoded.pts().unwrap() + delay.output));
                            }
                            // println!("resampled2 pts: {:?}", resampled.pts());
                            // println!("resampled planes: {}", resampled.planes());
                            if resampled.planes() == 0 {
                                // println!("break planes!");
                                break;
                            }
                            if resampled.samples() <= 0 {
                                // println!("break samples!");
                                break;
                            }

                            audio_tx.send(resampled).unwrap();
                            if result.is_none() {
                                // println!("break result!");
                                break;
                            }
                        }
                    }
                    // if result.is_some() {
                    //     let result = resampler.flush(&mut resampled).unwrap();
                    //     audio_tx.send(resampled).unwrap();
                    //     if result.is_none() {
                    //         break;
                    //     }
                    // }
                    // println!("result {:?}", result);

                    // println!("audio frame {:?}", decoded.timestamp());
                }
                Ok(())
            };

        // let position = 3.rescale((1, 1), rescale::TIME_BASE);
        // println!("position {:?}", position);
        // ictx.seek(position, ..position).unwrap();
        // // decoder.flush();
        // println!("first: {}", ictx.stream(0).unwrap().time_base().unwrap());
        // println!("second: {}", ictx.stream(1).unwrap().time_base().unwrap());
        // println!("seek");

        for res in ictx.packets() {
            let (stream, packet) = res.unwrap();
            if stream.index() == video_stream_index {
                video_decoder.send_packet(&packet).unwrap();
                receive_and_process_decoded_video_frames(&mut video_decoder).unwrap();
            } else if stream.index() == audio_stream_index {
                audio_decoder.send_packet(&packet).unwrap();
                receive_and_process_decoded_audio_frames(&mut audio_decoder).unwrap();
                // let mut decoded = effempeg::frame::Audio::empty();
                // while audio_decoder.receive_frame(&mut decoded).is_ok() {
                //     println!("audio frame {:?}", decoded.timestamp());
                // }
            }
        }
        video_decoder.send_eof().unwrap();
        receive_and_process_decoded_video_frames(&mut video_decoder).unwrap();
    });

    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            r"
// Vertex shader

struct VertexInput {
    @location(0) position: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
};
            
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = model.position;
    out.clip_position.a  = 1.0;
    out.vert_pos = out.clip_position.xyz;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct SceneUniform {
    width: f32,
    height: f32,
    vertexes: array<vec4<f32>, 4>,
};
@group(1) @binding(0)
var<uniform> scene: SceneUniform;


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // TODO: think this through
    let dim = textureDimensions(t_diffuse);
    if dim.x > dim.y {
        // Horizontal

        var logical_offset_x = (scene.vertexes[0].x + 1) / 2;
        var logical_offset_y = (scene.vertexes[0].y + 1) / 2;
        var logical_size_x =((scene.vertexes[2].x - scene.vertexes[0].x) / 2);
        var logical_size_y = (scene.vertexes[2].y - scene.vertexes[0].y) / 2;
        let actual_size_x = logical_size_x * scene.width;
        let actual_size_y = logical_size_y * scene.height;
        let actual_offset_x = logical_offset_x * scene.width;
        let actual_offset_y = logical_offset_y * scene.height;
        let ratio = f32(dim.y) / f32(dim.x);
        let final_size_y = ratio * actual_size_x;
        let actual_black_offset_y = (actual_size_y - final_size_y)/2;

        let x = (in.clip_position.x - actual_offset_x) / actual_size_x;
        let y = (in.clip_position.y - actual_offset_y - actual_black_offset_y) / final_size_y;

        if (y < 0 || y > 1) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        return textureSample(t_diffuse, s_diffuse, vec2<f32>(x, y));
    } else {
        // Vertical
        var logical_offset_x = (scene.vertexes[0].x + 1) / 2;
        var logical_offset_y = (scene.vertexes[0].y + 1) / 2;
        var logical_size_x =((scene.vertexes[2].x - scene.vertexes[0].x) / 2);
        var logical_size_y = (scene.vertexes[2].y - scene.vertexes[0].y) / 2;
        let actual_size_x = logical_size_x * scene.width;
        let actual_size_y = logical_size_y * scene.height;
        let actual_offset_x = logical_offset_x * scene.width;
        let actual_offset_y = logical_offset_y * scene.height;
        let ratio = f32(dim.x) / f32(dim.y);
        let final_size_x = ratio * actual_size_y;
        let actual_black_offset_x = (actual_size_x - final_size_x)/2;

        let x = (in.clip_position.x - actual_offset_x - actual_black_offset_x) / final_size_x;
        let y = (in.clip_position.y - actual_offset_y) / actual_size_y;

        if (x < 0 || x > 1) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        return textureSample(t_diffuse, s_diffuse, vec2<f32>(x, y));
    }
}
",
        )),
    });

    let mut scene: Scene = Scene {
        width: 100.0,
        height: 100.0,
        _padding: [0.0, 0.0],
        vertexes: VERTICES[0..4].try_into().unwrap(),
    };

    let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene Buffer"),
        contents: bytemuck::cast_slice(&[scene]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let scene_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("scene_bind_group_layout"),
        });

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &scene_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: scene_buffer.as_entire_binding(),
        }],
        label: Some("scene_bind_group"),
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&VERTICES),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let vertex_buffer_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x4,
        }],
    };

    // let vertex_bind_group_layout =
    //     device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //         entries: &[wgpu::BindGroupLayoutEntry {
    //             binding: 0,
    //             visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
    //             ty: wgpu::BindingType::Buffer {
    //                 ty: wgpu::BufferBindingType::Uniform,
    //                 has_dynamic_offset: false,
    //                 min_binding_size: None,
    //             },
    //             count: None,
    //         }],
    //         label: Some("vertex_bind_group_layout"),
    //     });

    // let vertex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //     layout: &scene_bind_group_layout,
    //     entries: &[wgpu::BindGroupEntry {
    //         binding: 0,
    //         resource: vertex_buffer.as_entire_binding(),
    //     }],
    //     label: Some("vertex_bind_group"),
    // });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    // let diffuse_bytes = include_bytes!("../tree.png");
    // let diffuse_bytes2 = include_bytes!("../code.png");
    // let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
    // let diffuse_image2 = image::load_from_memory(diffuse_bytes2).unwrap();
    // let diffuse_rgba = diffuse_image.to_rgba8();
    // let diffuse_rgba2 = diffuse_image2.to_rgba8();

    use image::GenericImageView;
    // let dimensions = diffuse_image.dimensions();

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
        // All textures are stored as 3D, we represent our 2D texture
        // by setting depth to 1.
        size: texture_size,
        mip_level_count: 1, // We'll talk about this a little later
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Most images are stored using sRGB, so we need to reflect that here.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
        // COPY_DST means that we want to copy data to this texture
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("diffuse_texture"),
        // This is the same as with the SurfaceConfig. It
        // specifies what texture formats can be used to
        // create TextureViews for this texture. The base
        // texture format (Rgba8UnormSrgb in this case) is
        // always supported. Note that using a different
        // texture format is not supported on the WebGL2
        // backend.
        view_formats: &[],
    });

    // queue.write_texture(
    //     // Tells wgpu where to copy the pixel data
    //     wgpu::ImageCopyTexture {
    //         texture: &diffuse_texture,
    //         mip_level: 0,
    //         origin: wgpu::Origin3d::ZERO,
    //         aspect: wgpu::TextureAspect::All,
    //     },
    //     // The actual pixel data
    //     &diffuse_rgba,
    //     // The layout of the texture
    //     wgpu::ImageDataLayout {
    //         offset: 0,
    //         bytes_per_row: Some(4 * dimensions.0),
    //         rows_per_image: Some(dimensions.1),
    //     },
    //     texture_size,
    // );

    // We don't need to configure the texture view much, so let's
    // let wgpu define it.
    let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        // min_filter: wgpu::FilterMode::Nearest,
        // mipmap_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // This should match the filterable field of the
                    // corresponding Texture entry above.
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

    let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
            },
        ],
        label: Some("diffuse_bind_group"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&texture_bind_group_layout, &scene_bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[vertex_buffer_layout],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut current = true;

    let window = &window;
    let event_loop_proxy = event_loop.create_proxy();

    std::thread::spawn(move || loop {
        event_loop_proxy.send_event(()).unwrap();

        // std::thread::sleep(std::time::Duration::from_millis(8));
        std::thread::sleep(std::time::Duration::from_millis(16));
        // std::thread::sleep(std::time::Duration::from_millis(32));
    });

    let mut last_instant = Instant::now();

    let mut sample_data: VecDeque<f32> = VecDeque::new();
    let mut output_data: Vec<f32> = Vec::new();

    let stream = audio_device
        .build_output_stream(
            &audio_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // println!(
                //     "data len {}, sample_data buffer {}",
                //     data.len(),
                //     sample_data.len()
                // );
                // for sample in data.iter_mut() {
                //     *sample = 0.0;
                // }
                // for chunk in data.chunks_mut(2) {
                //     let data_sample = if let Some(sample) = sample_data.pop_front() {
                //         sample
                //     } else {
                //         let frame = audio_receiver.recv().unwrap();
                //         let frame_data = frame.data(0);
                //         let frame_data_f32: &[f32] = unsafe {
                //             std::slice::from_raw_parts(
                //                 frame_data.as_ptr() as *const f32,
                //                 frame_data.len() / std::mem::size_of::<f32>(),
                //             )
                //         };

                //         println!("frame len: {}", frame_data_f32.len());

                //         sample_data.extend(frame_data_f32.iter());
                //         sample_data.pop_front().unwrap()
                //     };

                //     for sample in chunk {
                //         *sample = data_sample;
                //     }
                // }
                for out_sample in data.iter_mut() {
                    let data_sample = if let Some(sample) = sample_data.pop_front() {
                        sample
                    } else {
                        let frame = audio_receiver.recv().unwrap();

                        // let frame_timestamp_millis =
                        //     frame.pts().unwrap() / (audio_timebase.1 as i64 / 1000);
                        // // println!("frame {:?} {:?}", frame.pts(), frame_timestamp_millis);
                        // let actual_time_millis = last_instant.elapsed().as_millis() as i64;
                        // println!(
                        //     "delta audio: {}",
                        //     frame_timestamp_millis - actual_time_millis
                        // );

                        // println!("output frame planes {}", frame.planes());
                        let frame_data = frame.data(0);
                        let frame_data_f32: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                frame_data.as_ptr() as *const f32,
                                frame.samples() * frame.channels() as usize,
                            )
                        };

                        // println!("frame len: {}", frame_data_f32.len());

                        sample_data.extend(frame_data_f32.iter());
                        sample_data.pop_front().unwrap()
                    };

                    // if data_sample > 1.0 || data_sample < -1.0 {
                    //     println!("detected outlier!");
                    // }
                    // // Clip data sample to -0.5/0.5 range
                    // let data_sample = data_sample.max(-0.5).min(0.5);

                    *out_sample = data_sample;
                    output_data.push(data_sample);
                }

                // if output_data.len() > 200000 {
                //     let value = serde_json::Value::Array(
                //         output_data
                //             .iter()
                //             .map(|item| {
                //                 serde_json::Value::Number(
                //                     serde_json::Number::from_f64(*item as f64).unwrap_or_else(
                //                         || {
                //                             eprintln!("error converting to f64: {}", *item);
                //                             return serde_json::Number::from_f64(-1000000.0)
                //                                 .unwrap();
                //                         },
                //                     ),
                //                 )
                //             })
                //             .collect(),
                //     );

                //     std::fs::write("output.json", serde_json::to_string(&value).unwrap()).unwrap();
                //     panic!("done!");
                // }
            },
            move |err| {
                eprintln!("an error occurred on stream: {}", err);
            },
            None,
        )
        .unwrap();
    stream.play().unwrap();

    let mut last_video_frame = None;

    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &shader, &pipeline_layout);

            match event {
                Event::WindowEvent { window_id, event } => {
                    match event {
                        WindowEvent::Resized(new_size) => {
                            // Reconfigure the surface with the new size
                            config.width = new_size.width.max(1);
                            config.height = new_size.height.max(1);
                            surface.configure(&device, &config);
                            // On macos the window needs to be redrawn manually after resizing
                            window.request_redraw();
                        }
                        WindowEvent::RedrawRequested => {
                            let inner_size = window.inner_size();
                            scene = Scene {
                                width: inner_size.width as f32,
                                height: inner_size.height as f32,
                                _padding: [0.0, 0.0],
                                vertexes: VERTICES[0..4].try_into().unwrap(),
                            };
                            queue.write_buffer(&scene_buffer, 0, bytemuck::cast_slice(&[scene]));
                            // println!("scene {:?}", scene);
                            // VERTICES[0].position[0] += 0.01;
                            // queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&VERTICES));

                            let frame = loop {
                                let frame: Video =
                                    if let Some(last_video_frame) = last_video_frame.take() {
                                        last_video_frame
                                    } else {
                                        video_receiver.recv().unwrap()
                                    };
                                let frame_timestamp_millis = frame.pts().unwrap() as f64
                                    / (video_timebase.1 as f64 / 1000.0);
                                let frame_timestamp_millis = frame_timestamp_millis as i64;
                                // println!("frame {:?} {:?}", frame.pts(), frame_timestamp_millis);
                                let actual_time_millis = last_instant.elapsed().as_millis() as i64;

                                // println!(
                                //     "delta video: {}",
                                //     frame_timestamp_millis - actual_time_millis
                                // );

                                if frame_timestamp_millis > actual_time_millis {
                                    // println!("delaying frame");
                                    last_video_frame = Some(frame);
                                    return;
                                } else if frame_timestamp_millis < actual_time_millis - 16 {
                                    println!("skipping frame");
                                    last_video_frame = None;
                                    continue;
                                }

                                break frame;
                            };

                            let bytes = frame.data(0);

                            current = !current;
                            queue.write_texture(
                                // Tells wgpu where to copy the pixel data
                                wgpu::ImageCopyTexture {
                                    texture: &diffuse_texture,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                // The actual pixel data
                                &bytes,
                                // The layout of the texture
                                wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(4 * width),
                                    rows_per_image: Some(height),
                                },
                                texture_size,
                            );

                            let frame = surface
                                .get_current_texture()
                                .expect("Failed to acquire next swap chain texture");
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });
                            {
                                let mut rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                rpass.set_pipeline(&render_pipeline);
                                rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                                rpass.set_index_buffer(
                                    index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint16,
                                );
                                rpass.set_bind_group(0, &diffuse_bind_group, &[]);
                                rpass.set_bind_group(1, &scene_bind_group, &[]);
                                rpass.draw_indexed(0..(INDICES.len() as u32), 0, 0..1);
                            }

                            queue.submit(Some(encoder.finish()));
                            frame.present();
                        }
                        WindowEvent::CloseRequested => target.exit(),
                        _ => {}
                    };
                }
                Event::UserEvent(_) => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut builder = winit::window::WindowBuilder::new().with_inner_size(Size::Physical(
        winit::dpi::PhysicalSize {
            height: 600,
            width: 600,
        },
    ));
    let window = builder.build(&event_loop).unwrap();

    pollster::block_on(run(event_loop, window));
}
