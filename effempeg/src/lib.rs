#![allow(non_camel_case_types)]
#![cfg_attr(feature = "cargo-clippy", allow(inline_always))]

#[macro_use]
extern crate bitflags;
pub extern crate effempeg_sys as sys;
#[cfg(feature = "image")]
extern crate image;

pub use effempeg_sys as ffi;

#[macro_use]
pub mod util;
pub use crate::util::{
    channel_layout::{self, Channel, ChannelLayout},
    chroma, color, dictionary,
    dictionary::{Mut as DictionaryMut, Owned as Dictionary, Ref as DictionaryRef},
    error::Error,
    frame::{self, Frame},
    mathematics::{self, rescale, Rescale, Rounding},
    media, option, picture,
    rational::{self, Rational},
    time,
};

pub type Result<T> = ::std::result::Result<T, Error>;

pub mod format;

pub use crate::format::chapter::{Chapter, ChapterMut};

pub use crate::format::format::Format;

pub use crate::format::stream::{Stream, StreamMut};

pub mod codec;

pub use crate::codec::audio_service::AudioService;

pub use crate::codec::codec::Codec;

pub use crate::codec::discard::Discard;

pub use crate::codec::field_order::FieldOrder;

pub use crate::codec::packet::{self, Packet};

pub use crate::codec::subtitle::{self, Subtitle};

pub use crate::codec::threading;

pub use crate::codec::{decoder, encoder};

pub mod device;

pub mod filter;
pub use crate::filter::Filter;

pub mod software;

fn init_util() {
    util::error::register_all();
    #[cfg(feature = "log")]
    util::log::register();
}

fn init_device() {
    device::register_all();
}

pub fn init() -> Result<()> {
    init_util();
    init_device();

    Ok(())
}
