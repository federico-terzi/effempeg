use std::ptr;

use libc::c_int;

use super::Disposition;
use crate::{
    channel_layout::ChannelOrder,
    codec::{self, packet, Decoder},
    ffi::*,
    format::context::common::Context,
    ChannelLayout, DictionaryRef, Discard, Error, Rational,
};

#[derive(Debug)]
pub struct Stream<'a> {
    context: &'a Context,
    index: usize,
}

impl<'a> Stream<'a> {
    pub unsafe fn wrap(context: &Context, index: usize) -> Stream<'_> {
        Stream { context, index }
    }

    pub unsafe fn as_ptr(&self) -> *const AVStream {
        *(*self.context.as_ptr()).streams.add(self.index)
    }
}

impl<'a> Stream<'a> {
    pub fn id(&self) -> i32 {
        unsafe { (*self.as_ptr()).id }
    }

    // pub fn codec(&self) -> codec::Context {
    //     unsafe { codec::Context::wrap((*self.as_ptr()).codec, Some(self.context.destructor())) }
    // }

    pub fn parameters(&self) -> codec::Parameters {
        unsafe {
            codec::Parameters::wrap((*self.as_ptr()).codecpar, Some(self.context.destructor()))
        }

        // #[cfg(not(feature = "ffmpeg_3_1"))]
        // unsafe {
        //     codec::Parameters::wrap((*self.as_ptr()).codec, Some(self.context.destructor()))
        // }
    }

    pub fn index(&self) -> usize {
        unsafe { (*self.as_ptr()).index as usize }
    }

    pub fn time_base(&self) -> Option<Rational> {
        unsafe { Rational::from((*self.as_ptr()).time_base).non_zero() }
    }

    pub fn start_time(&self) -> Option<i64> {
        unsafe {
            match (*self.as_ptr()).start_time {
                AV_NOPTS_VALUE => None,
                value => Some(value),
            }
        }
    }

    pub fn duration(&self) -> Option<i64> {
        unsafe {
            match (*self.as_ptr()).duration {
                AV_NOPTS_VALUE => None,
                value => Some(value),
            }
        }
    }

    pub fn frames(&self) -> i64 {
        unsafe { (*self.as_ptr()).nb_frames }
    }

    pub fn disposition(&self) -> Disposition {
        unsafe { Disposition::from_bits_truncate((*self.as_ptr()).disposition) }
    }

    pub fn discard(&self) -> Discard {
        unsafe { Discard::from((*self.as_ptr()).discard) }
    }

    pub fn side_data(&self) -> SideDataIter<'_> {
        SideDataIter::new(self)
    }

    pub fn frame_rate(&self) -> Rational {
        unsafe { Rational::from((*self.as_ptr()).r_frame_rate) }
    }

    pub fn avg_frame_rate(&self) -> Rational {
        unsafe { Rational::from((*self.as_ptr()).avg_frame_rate) }
    }

    pub fn guess_frame_rate(&self) -> Option<Rational> {
        unsafe {
            let r = Rational::from(av_guess_frame_rate(
                self.context.as_ptr() as *mut _,
                self.as_ptr() as *mut _,
                ptr::null_mut(),
            ));

            r.non_zero()
        }
    }

    pub fn metadata(&self) -> DictionaryRef<'_> {
        unsafe { DictionaryRef::wrap((*self.as_ptr()).metadata) }
    }

    pub fn display_rotation(&self) -> f64 {
        if let Some(display_matrix) = self
            .side_data()
            .find(|d| d.kind() == packet::side_data::Type::DisplayMatrix)
        {
            unsafe {
                let bytes = display_matrix.data();
                let (prefix, matrix, suffix) = bytes.align_to::<i32>();
                // data result should be aligned &[i32; 9]
                assert!(prefix.is_empty() && suffix.is_empty() && matrix.len() == 9);
                av_display_rotation_get(matrix.as_ptr())
            }
        } else {
            0.0f64
        }
    }

    pub fn decoder(&self) -> Result<Decoder, Error> {
        let params = self.parameters();
        let codec = codec::decoder::find(params.id()).ok_or(Error::DecoderNotFound)?;
        let mut dec = Decoder::new(codec)?;

        dec.set_parameters(params)?;
        dec.set_time_base(self.time_base());
        dec.set_frame_rate(self.guess_frame_rate());

        // Sometimes the decoder is opened with unspecified channel layout,
        // and that causes the resampler to fail.
        // So I'm setting the layout as in this example: https://ffmpeg.org/doxygen/6.0/decode_filter_audio_8c-example.html
        if dec.channel_layout().order() == ChannelOrder::Unspecified {
            let channel_layout = ChannelLayout::default(dec.channel_layout().channels());
            dec.set_channel_layout(channel_layout);
        }

        Ok(dec)
    }
}

impl<'a> PartialEq for Stream<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.as_ptr() == other.as_ptr() }
    }
}

impl<'a> Eq for Stream<'a> {}

pub struct SideDataIter<'a> {
    stream: &'a Stream<'a>,
    current: c_int,
}

impl<'a> SideDataIter<'a> {
    pub fn new<'sd, 's: 'sd>(stream: &'s Stream<'_>) -> SideDataIter<'sd> {
        SideDataIter { stream, current: 0 }
    }
}

impl<'a> Iterator for SideDataIter<'a> {
    type Item = packet::SideData<'a>;

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        unsafe {
            if self.current >= (*self.stream.as_ptr()).nb_side_data {
                return None;
            }

            self.current += 1;

            Some(packet::SideData::wrap(
                (*self.stream.as_ptr())
                    .side_data
                    .offset((self.current - 1) as isize),
            ))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let length = (*self.stream.as_ptr()).nb_side_data as usize;

            (
                length - self.current as usize,
                Some(length - self.current as usize),
            )
        }
    }
}

impl<'a> ExactSizeIterator for SideDataIter<'a> {}
