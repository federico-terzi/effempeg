use std::ptr;

use libc::c_int;

use super::Delay;
use crate::{
    ffi::*,
    frame,
    util::{format, mathematics::rescale::TIME_BASE},
    ChannelLayout, Error,
};

#[derive(Copy, Clone)]
pub struct Definition {
    pub format: format::Sample,
    pub channel_layout: ChannelLayout,
    pub rate: u32,
}

#[derive(Clone)]
pub struct ResamplingMatrix {
    pub matrix: Vec<f64>,
    pub stride: i32,
}

pub struct Context {
    ptr: *mut SwrContext,

    input: Definition,
    output: Definition,
    matrix: Option<ResamplingMatrix>,
}

unsafe impl Send for Context {}

impl Context {
    #[doc(hidden)]
    pub unsafe fn as_ptr(&self) -> *const SwrContext {
        self.ptr as *const _
    }

    #[doc(hidden)]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut SwrContext {
        self.ptr
    }
}

impl Context {
    /// Create a resampler with the given definitions.
    pub fn get(
        src_format: format::Sample,
        src_channel_layout: ChannelLayout,
        src_rate: u32,
        dst_format: format::Sample,
        dst_channel_layout: ChannelLayout,
        dst_rate: u32
    ) -> Result<Self, Error> {
        Self::get_with_matrix(src_format, src_channel_layout, src_rate, dst_format, dst_channel_layout, dst_rate, None)
    }

    /// Create a resampler with the given definitions.
    pub fn get_with_matrix(
        src_format: format::Sample,
        src_channel_layout: ChannelLayout,
        src_rate: u32,
        dst_format: format::Sample,
        dst_channel_layout: ChannelLayout,
        dst_rate: u32,
        matrix: Option<ResamplingMatrix>,
    ) -> Result<Self, Error> {
        unsafe {
            let mut ctx = ptr::null_mut::<SwrContext>();

            // #[cfg(not(feature = "ffmpeg_5_1"))]
            // {
            // 	ctx = swr_alloc_set_opts(
            // 		ptr::null_mut(),
            // 		dst_channel_layout.bits() as i64,
            // 		dst_format.into(),
            // 		dst_rate as c_int,
            // 		src_channel_layout.bits() as i64,
            // 		src_format.into(),
            // 		src_rate as c_int,
            // 		0,
            // 		ptr::null_mut(),
            // 	);
            // }

            {
                swr_alloc_set_opts2(
                    &mut ctx as *mut _,
                    dst_channel_layout.as_ptr(),
                    dst_format.into(),
                    dst_rate as c_int,
                    src_channel_layout.as_ptr(),
                    src_format.into(),
                    src_rate as c_int,
                    0,
                    ptr::null_mut(),
                );
            }

            if !ctx.is_null() {
                if let Some(matrix) = &matrix {
                    swr_set_matrix(ctx, matrix.matrix.as_ptr(), matrix.stride);
                }

                match swr_init(ctx) {
                    e if e < 0 => Err(Error::from(e)),

                    _ => Ok(Context {
                        ptr: ctx,

                        input: Definition {
                            format: src_format,
                            channel_layout: src_channel_layout,
                            rate: src_rate,
                        },

                        output: Definition {
                            format: dst_format,
                            channel_layout: dst_channel_layout,
                            rate: dst_rate,
                        },

                        matrix,
                    }),
                }
            } else {
                Err(Error::InvalidData)
            }
        }
    }

    /// Get the input definition.
    pub fn input(&self) -> &Definition {
        &self.input
    }

    /// Get the output definition.
    pub fn output(&self) -> &Definition {
        &self.output
    }

    /// Get the remaining delay.
    pub fn delay(&self) -> Option<Delay> {
        unsafe {
            match swr_get_delay(self.as_ptr() as *mut _, TIME_BASE.1 as i64) {
                0 => None,
                _ => Some(Delay::from(self)),
            }
        }
    }

    /// Run the resampler from the given input to the given output.
    ///
    /// When there are internal frames to process it will return `Ok(Some(Delay {
    /// .. }))`.
    pub fn run(
        &mut self,
        input: &frame::Audio,
        output: &mut frame::Audio,
    ) -> Result<Option<Delay>, Error> {
        output.set_rate(self.output.rate);

        unsafe {
            if output.is_empty() {
                output.alloc(
                    self.output.format,
                    input.samples(),
                    self.output.channel_layout,
                );
            }

            match swr_convert_frame(self.as_mut_ptr(), output.as_mut_ptr(), input.as_ptr()) {
                0 => Ok(self.delay()),

                e => Err(Error::from(e)),
            }
        }
    }

    /// Convert one of the remaining internal frames.
    ///
    /// When there are no more internal frames `Ok(None)` will be returned.
    pub fn flush(&mut self, output: &mut frame::Audio) -> Result<Option<Delay>, Error> {
        output.set_rate(self.output.rate);

        unsafe {
            match swr_convert_frame(self.as_mut_ptr(), output.as_mut_ptr(), ptr::null()) {
                0 => Ok(self.delay()),

                e => Err(Error::from(e)),
            }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            swr_free(&mut self.as_mut_ptr());
        }
    }
}
