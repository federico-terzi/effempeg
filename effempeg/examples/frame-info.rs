use std::{env, fs::File, io::prelude::*, process::exit};

use effempeg::{
    format::{input, Pixel},
    media::Type,
    software::scaling::{context::Context, flag::Flags},
    util::frame::video::Video,
};

fn main() -> Result<(), effempeg::Error> {
    effempeg::init().unwrap();

    if let Ok(mut ictx) = input(&env::args().nth(1).expect("Cannot open file.")) {
        let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or(effempeg::Error::StreamNotFound)?;
        let video_stream_index = input.index();

        let mut decoder = input.decoder()?.video()?;

        let mut scaler = Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::ARGB,
            decoder.width(),
            decoder.height(),
            Flags::BILINEAR,
        )?;

        // let mut frame_index = 0;

        let mut receive_and_process_decoded_frames =
            |decoder: &mut effempeg::decoder::Video| -> Result<(), effempeg::Error> {
                let mut decoded = Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut converted_frame = Video::empty();
                    scaler.run(&decoded, &mut converted_frame)?;
                    println!("Frame: {:?}", converted_frame.format());
                    let native_frame = converted_frame.0.as_ptr();

                    unsafe {
                        let format = (*native_frame).format;
                        let width = (*native_frame).width;
                        let height = (*native_frame).height;
                        let linesize = (*native_frame).linesize;
                        let data = (*native_frame).data;
                        let nb_samples = (*native_frame).nb_samples;
                        let key_frame = (*native_frame).key_frame;
                        let buf = (*native_frame).buf;

                        let second_plane_diff = data[1] as i64 - data[0] as i64;

                        dbg!(
                            format,
                            width,
                            height,
                            linesize,
                            data,
                            second_plane_diff,
                            nb_samples,
                            key_frame,
                            buf
                        );
                    }
                    // save_file(&rgb_frame, frame_index).unwrap();
                    // frame_index += 1;
                    exit(0);
                }
                Ok(())
            };

        for res in ictx.packets() {
            let (stream, packet) = res?;
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)?;
                receive_and_process_decoded_frames(&mut decoder)?;
            }
        }
        decoder.send_eof()?;
        receive_and_process_decoded_frames(&mut decoder)?;
    }

    Ok(())
}
