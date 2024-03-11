// Derived from https://github.com/zmwangx/rust-ffmpeg-sys

extern crate bindgen;

use std::{env, path::PathBuf};

use bindgen::callbacks::{
    EnumVariantCustomBehavior, EnumVariantValue, IntKind, MacroParsingBehavior, ParseCallbacks,
};

#[derive(Debug)]
struct Callbacks;

impl ParseCallbacks for Callbacks {
    fn int_macro(&self, _name: &str, value: i64) -> Option<IntKind> {
        let ch_layout_prefix = "AV_CH_";
        let codec_cap_prefix = "AV_CODEC_CAP_";
        let codec_flag_prefix = "AV_CODEC_FLAG_";
        let error_max_size = "AV_ERROR_MAX_STRING_SIZE";

        if value >= i64::min_value() && _name.starts_with(ch_layout_prefix) {
            Some(IntKind::ULongLong)
        } else if value >= i32::min_value() as i64
            && value <= i32::max_value() as i64
            && (_name.starts_with(codec_cap_prefix) || _name.starts_with(codec_flag_prefix))
        {
            Some(IntKind::UInt)
        } else if _name == error_max_size {
            Some(IntKind::Custom {
                name: "usize",
                is_signed: false,
            })
        } else if value >= i32::min_value() as i64 && value <= i32::max_value() as i64 {
            Some(IntKind::Int)
        } else {
            None
        }
    }

    fn enum_variant_behavior(
        &self,
        _enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<EnumVariantCustomBehavior> {
        let dummy_codec_id_prefix = "AV_CODEC_ID_FIRST_";
        if original_variant_name.starts_with(dummy_codec_id_prefix) {
            Some(EnumVariantCustomBehavior::Constify)
        } else {
            None
        }
    }

    // https://github.com/rust-lang/rust-bindgen/issues/687#issuecomment-388277405
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        use MacroParsingBehavior::*;

        match name {
            "FP_INFINITE" => Ignore,
            "FP_NAN" => Ignore,
            "FP_NORMAL" => Ignore,
            "FP_SUBNORMAL" => Ignore,
            "FP_ZERO" => Ignore,
            _ => Default,
        }
    }
}

fn get_ffmpeg_path() -> PathBuf {
    let manifest_path = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = PathBuf::from(manifest_path);
    let parent_path = manifest_path.parent().unwrap();
    let ffmpeg_path = parent_path.join("FFmpeg");
    if !ffmpeg_path.exists() {
        panic!("FFmpeg source code not found at {:?}", ffmpeg_path);
    }

    ffmpeg_path
}

fn get_header_path(header: &str) -> String {
    let parts = header.split('/').collect::<Vec<_>>();
    let mut path = PathBuf::new();
    path.push(get_ffmpeg_path());
    path.extend(parts);

    path.to_string_lossy().to_string()
}

fn main() {
    let include_paths = vec![get_ffmpeg_path().join("INCLUDES")];
    let clang_includes: Vec<String> = include_paths
        .iter()
        .map(|include| format!("-I{}", include.to_string_lossy()))
        .collect();

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut builder = bindgen::Builder::default()
        .clang_args(clang_includes)
        .ctypes_prefix("libc")
        // https://github.com/rust-lang/rust-bindgen/issues/550
        .blocklist_type("max_align_t")
        // Issue on aligned and packed struct. Related to:
        // https://github.com/rust-lang/rust-bindgen/issues/1538
        .opaque_type("__mingw_ldbl_type_t")
        // these are never part of ffmpeg API
        .blocklist_function("_.*")
        // Rust doesn't support long double, and bindgen can't skip it
        // https://github.com/rust-lang/rust-bindgen/issues/1549
        .blocklist_function("acoshl")
        .blocklist_function("acosl")
        .blocklist_function("asinhl")
        .blocklist_function("asinl")
        .blocklist_function("atan2l")
        .blocklist_function("atanhl")
        .blocklist_function("atanl")
        .blocklist_function("cbrtl")
        .blocklist_function("ceill")
        .blocklist_function("copysignl")
        .blocklist_function("coshl")
        .blocklist_function("cosl")
        .blocklist_function("dreml")
        .blocklist_function("ecvt_r")
        .blocklist_function("erfcl")
        .blocklist_function("erfl")
        .blocklist_function("exp2l")
        .blocklist_function("expl")
        .blocklist_function("expm1l")
        .blocklist_function("fabsl")
        .blocklist_function("fcvt_r")
        .blocklist_function("fdiml")
        .blocklist_function("finitel")
        .blocklist_function("floorl")
        .blocklist_function("fmal")
        .blocklist_function("fmaxl")
        .blocklist_function("fminl")
        .blocklist_function("fmodl")
        .blocklist_function("frexpl")
        .blocklist_function("gammal")
        .blocklist_function("hypotl")
        .blocklist_function("ilogbl")
        .blocklist_function("isinfl")
        .blocklist_function("isnanl")
        .blocklist_function("j0l")
        .blocklist_function("j1l")
        .blocklist_function("jnl")
        .blocklist_function("ldexpl")
        .blocklist_function("lgammal")
        .blocklist_function("lgammal_r")
        .blocklist_function("llrintl")
        .blocklist_function("llroundl")
        .blocklist_function("log10l")
        .blocklist_function("log1pl")
        .blocklist_function("log2l")
        .blocklist_function("logbl")
        .blocklist_function("logl")
        .blocklist_function("lrintl")
        .blocklist_function("lroundl")
        .blocklist_function("modfl")
        .blocklist_function("nanl")
        .blocklist_function("nearbyintl")
        .blocklist_function("nextafterl")
        .blocklist_function("nexttoward")
        .blocklist_function("nexttowardf")
        .blocklist_function("nexttowardl")
        .blocklist_function("powl")
        .blocklist_function("qecvt")
        .blocklist_function("qecvt_r")
        .blocklist_function("qfcvt")
        .blocklist_function("qfcvt_r")
        .blocklist_function("qgcvt")
        .blocklist_function("remainderl")
        .blocklist_function("remquol")
        .blocklist_function("rintl")
        .blocklist_function("roundl")
        .blocklist_function("scalbl")
        .blocklist_function("scalblnl")
        .blocklist_function("scalbnl")
        .blocklist_function("significandl")
        .blocklist_function("sinhl")
        .blocklist_function("sinl")
        .blocklist_function("sqrtl")
        .blocklist_function("strtold")
        .blocklist_function("tanhl")
        .blocklist_function("tanl")
        .blocklist_function("tgammal")
        .blocklist_function("truncl")
        .blocklist_function("y0l")
        .blocklist_function("y1l")
        .blocklist_function("ynl")
        .newtype_enum("AVChannel")
        .newtype_enum("AVChannelOrder")
        .rustified_enum(".*")
        .prepend_enum_name(false)
        .derive_eq(true)
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(Callbacks));

    // TODO: parametrize this properly
    builder = builder
        .header(get_header_path("libavcodec/avcodec.h"))
        .header(get_header_path("libavcodec/dv_profile.h"))
        .header(get_header_path("libavcodec/avfft.h"))
        .header(get_header_path("libavcodec/vorbis_parser.h"));

    builder = builder.header(get_header_path("libavdevice/avdevice.h"));

    builder = builder
        .header(get_header_path("libavfilter/buffersink.h"))
        .header(get_header_path("libavfilter/buffersrc.h"))
        .header(get_header_path("libavfilter/avfilter.h"));

    builder = builder
        .header(get_header_path("libavformat/avformat.h"))
        .header(get_header_path("libavformat/avio.h"));

    // builder = builder.header(get_header_path("libavresample/avresample.h"));

    builder = builder
        .header(get_header_path("libavutil/adler32.h"))
        .header(get_header_path("libavutil/aes.h"))
        .header(get_header_path("libavutil/audio_fifo.h"))
        .header(get_header_path("libavutil/base64.h"))
        .header(get_header_path("libavutil/blowfish.h"))
        .header(get_header_path("libavutil/bprint.h"))
        .header(get_header_path("libavutil/buffer.h"))
        .header(get_header_path("libavutil/camellia.h"))
        .header(get_header_path("libavutil/cast5.h"))
        .header(get_header_path("libavutil/channel_layout.h"))
        .header(get_header_path("libavutil/cpu.h"))
        .header(get_header_path("libavutil/crc.h"))
        .header(get_header_path("libavutil/dict.h"))
        .header(get_header_path("libavutil/display.h"))
        .header(get_header_path("libavutil/downmix_info.h"))
        .header(get_header_path("libavutil/error.h"))
        .header(get_header_path("libavutil/eval.h"))
        .header(get_header_path("libavutil/fifo.h"))
        .header(get_header_path("libavutil/file.h"))
        .header(get_header_path("libavutil/frame.h"))
        .header(get_header_path("libavutil/hash.h"))
        .header(get_header_path("libavutil/hmac.h"))
        .header(get_header_path("libavutil/hwcontext.h"))
        .header(get_header_path("libavutil/imgutils.h"))
        .header(get_header_path("libavutil/lfg.h"))
        .header(get_header_path("libavutil/log.h"))
        .header(get_header_path("libavutil/lzo.h"))
        .header(get_header_path("libavutil/macros.h"))
        .header(get_header_path("libavutil/mathematics.h"))
        .header(get_header_path("libavutil/md5.h"))
        .header(get_header_path("libavutil/mem.h"))
        .header(get_header_path("libavutil/motion_vector.h"))
        .header(get_header_path("libavutil/murmur3.h"))
        .header(get_header_path("libavutil/opt.h"))
        .header(get_header_path("libavutil/parseutils.h"))
        .header(get_header_path("libavutil/pixdesc.h"))
        .header(get_header_path("libavutil/pixfmt.h"))
        .header(get_header_path("libavutil/random_seed.h"))
        .header(get_header_path("libavutil/rational.h"))
        .header(get_header_path("libavutil/replaygain.h"))
        .header(get_header_path("libavutil/ripemd.h"))
        .header(get_header_path("libavutil/samplefmt.h"))
        .header(get_header_path("libavutil/sha.h"))
        .header(get_header_path("libavutil/sha512.h"))
        .header(get_header_path("libavutil/stereo3d.h"))
        .header(get_header_path("libavutil/avstring.h"))
        .header(get_header_path("libavutil/threadmessage.h"))
        .header(get_header_path("libavutil/time.h"))
        .header(get_header_path("libavutil/timecode.h"))
        .header(get_header_path("libavutil/twofish.h"))
        .header(get_header_path("libavutil/avutil.h"))
        .header(get_header_path("libavutil/xtea.h"));

    // builder = builder.header(get_header_path("libpostproc/postprocess.h"));

    builder = builder.header(get_header_path("libswresample/swresample.h"));

    builder = builder.header(get_header_path("libswscale/swscale.h"));

    // if let Some(hwcontext_drm_header) =
    //     maybe_search_include(&include_paths, "libavutil/hwcontext_drm.h")
    // {
    //     builder = builder.header(hwcontext_drm_header);
    // }

    // Finish the builder and generate the bindings.
    let bindings = builder
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("bindings.rs")
        .expect("Couldn't write bindings!");
}
