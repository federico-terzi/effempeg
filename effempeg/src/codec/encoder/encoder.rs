use std::{
	ops::{Deref, DerefMut},
	os::raw::c_int,
	ptr,
};

use super::{audio, subtitle, video};
use crate::{
	codec::{traits, Context},
	ffi::*,
	media, packet, Error, Frame,
};

pub struct Encoder(pub Context);

impl Encoder {
	pub fn new<E: traits::Encoder>(codec: E) -> Result<Self, Error> {
		let codec = codec.encoder().ok_or(Error::EncoderNotFound)?;
		let context = Context::for_codec(&codec);

		Ok(Encoder(context))
	}

	pub fn video(mut self) -> Result<video::Video, Error> {
		match self.medium() {
			media::Type::Unknown => {
				unsafe {
					(*self.as_mut_ptr()).codec_type = media::Type::Video.into();
				}

				Ok(video::Video(self))
			}

			media::Type::Video => Ok(video::Video(self)),

			_ => Err(Error::InvalidData),
		}
	}

	pub fn audio(mut self) -> Result<audio::Audio, Error> {
		match self.medium() {
			media::Type::Unknown => {
				unsafe {
					(*self.as_mut_ptr()).codec_type = media::Type::Audio.into();
				}

				Ok(audio::Audio(self))
			}

			media::Type::Audio => Ok(audio::Audio(self)),

			_ => Err(Error::InvalidData),
		}
	}

	pub fn subtitle(mut self) -> Result<subtitle::Subtitle, Error> {
		match self.medium() {
			media::Type::Unknown => {
				unsafe {
					(*self.as_mut_ptr()).codec_type = media::Type::Subtitle.into();
				}

				Ok(subtitle::Subtitle(self))
			}

			media::Type::Subtitle => Ok(subtitle::Subtitle(self)),

			_ => Err(Error::InvalidData),
		}
	}

	pub fn send_frame(&mut self, frame: &Frame) -> Result<(), Error> {
		unsafe {
			match avcodec_send_frame(self.as_mut_ptr(), frame.as_ptr()) {
				e if e < 0 => Err(Error::from(e)),
				_ => Ok(()),
			}
		}
	}

	/// Sends a NULL packet to the encoder to signal end of stream and enter
	/// draining mode.
	pub fn send_eof(&mut self) -> Result<(), Error> {
		unsafe { self.send_frame(&Frame::wrap(ptr::null_mut())) }
	}

	pub fn receive_packet<P: packet::Mut>(&mut self, packet: &mut P) -> Result<(), Error> {
		unsafe {
			match avcodec_receive_packet(self.as_mut_ptr(), packet.as_mut_ptr()) {
				e if e < 0 => Err(Error::from(e)),
				_ => Ok(()),
			}
		}
	}

	pub fn set_strict_std_compliance(&mut self, value: i32) {
		unsafe {
			(*self.as_mut_ptr()).strict_std_compliance = value as c_int;
		}
	}

	pub fn set_bit_rate(&mut self, value: usize) {
		unsafe {
			(*self.as_mut_ptr()).bit_rate = value as i64;
		}
	}

	pub fn set_max_bit_rate(&mut self, value: usize) {
		unsafe {
			(*self.as_mut_ptr()).rc_max_rate = value as i64;
		}
	}

	pub fn set_tolerance(&mut self, value: usize) {
		unsafe {
			(*self.as_mut_ptr()).bit_rate_tolerance = value as c_int;
		}
	}

	pub fn set_quality(&mut self, value: usize) {
		unsafe {
			(*self.as_mut_ptr()).global_quality = value as c_int;
		}
	}

	pub fn set_compression(&mut self, value: Option<usize>) {
		unsafe {
			if let Some(value) = value {
				(*self.as_mut_ptr()).compression_level = value as c_int;
			}
			else {
				(*self.as_mut_ptr()).compression_level = -1;
			}
		}
	}
}

impl Deref for Encoder {
	type Target = Context;

	fn deref(&self) -> &<Self as Deref>::Target {
		&self.0
	}
}

impl DerefMut for Encoder {
	fn deref_mut(&mut self) -> &mut <Self as Deref>::Target {
		&mut self.0
	}
}

impl AsRef<Context> for Encoder {
	fn as_ref(&self) -> &Context {
		self
	}
}

impl AsMut<Context> for Encoder {
	fn as_mut(&mut self) -> &mut Context {
		&mut *self
	}
}
