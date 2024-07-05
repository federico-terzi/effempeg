use std::{
    error,
    ffi::CStr,
    fmt,
    hash::{Hash, Hasher},
    io,
    os::raw::c_char,
    str::from_utf8_unchecked,
};

use libc::c_int;

use crate::ffi::*;

pub enum Error {
    Io(io::Error),
    Bug,
    Bug2,
    Unknown,
    Experimental,
    BufferTooSmall,
    Eof,
    Exit,
    External,
    InvalidData,
    PatchWelcome,

    Again,

    InputChanged,
    OutputChanged,

    BsfNotFound,
    DecoderNotFound,
    DemuxerNotFound,
    EncoderNotFound,
    OptionNotFound,
    MuxerNotFound,
    FilterNotFound,
    ProtocolNotFound,
    StreamNotFound,

    HttpBadRequest,
    HttpUnauthorized,
    HttpForbidden,
    HttpNotFound,
    HttpOther4xx,
    HttpServerError,
}

impl From<c_int> for Error {
    fn from(value: c_int) -> Error {
        match value {
            AVERROR_EAGAIN => Error::Again,
            AVERROR_BSF_NOT_FOUND => Error::BsfNotFound,
            AVERROR_BUG => Error::Bug,
            AVERROR_BUFFER_TOO_SMALL => Error::BufferTooSmall,
            AVERROR_DECODER_NOT_FOUND => Error::DecoderNotFound,
            AVERROR_DEMUXER_NOT_FOUND => Error::DemuxerNotFound,
            AVERROR_ENCODER_NOT_FOUND => Error::EncoderNotFound,
            AVERROR_EOF => Error::Eof,
            AVERROR_EXIT => Error::Exit,
            AVERROR_EXTERNAL => Error::External,
            AVERROR_FILTER_NOT_FOUND => Error::FilterNotFound,
            AVERROR_INVALIDDATA => Error::InvalidData,
            AVERROR_MUXER_NOT_FOUND => Error::MuxerNotFound,
            AVERROR_OPTION_NOT_FOUND => Error::OptionNotFound,
            AVERROR_PATCHWELCOME => Error::PatchWelcome,
            AVERROR_PROTOCOL_NOT_FOUND => Error::ProtocolNotFound,
            AVERROR_STREAM_NOT_FOUND => Error::StreamNotFound,
            AVERROR_BUG2 => Error::Bug2,
            AVERROR_UNKNOWN => Error::Unknown,
            AVERROR_EXPERIMENTAL => Error::Experimental,
            AVERROR_INPUT_CHANGED => Error::InputChanged,
            AVERROR_OUTPUT_CHANGED => Error::OutputChanged,
            AVERROR_HTTP_BAD_REQUEST => Error::HttpBadRequest,
            AVERROR_HTTP_UNAUTHORIZED => Error::HttpUnauthorized,
            AVERROR_HTTP_FORBIDDEN => Error::HttpForbidden,
            AVERROR_HTTP_NOT_FOUND => Error::HttpNotFound,
            AVERROR_HTTP_OTHER_4XX => Error::HttpOther4xx,
            AVERROR_HTTP_SERVER_ERROR => Error::HttpServerError,

            err => Error::Io(io::Error::from_raw_os_error(-err)),
        }
    }
}

impl Into<c_int> for Error {
    fn into(self) -> c_int {
        self.as_raw_error()
    }
}

impl From<Error> for io::Error {
    fn from(value: Error) -> io::Error {
        match value {
            Error::Io(err) => err,

            value => io::Error::new(io::ErrorKind::Other, value),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let description = match self {
            Error::Io(io) => format!("{:?}", io),
            Error::Again => "Again".to_string(),
            Error::Bug => "Bug".to_string(),
            Error::Bug2 => "Bug2".to_string(),
            Error::Unknown => "Unknown".to_string(),
            Error::Experimental => "Experimental".to_string(),
            Error::BufferTooSmall => "BufferTooSmall".to_string(),
            Error::Eof => "Eof".to_string(),
            Error::Exit => "Exit".to_string(),
            Error::External => "External".to_string(),
            Error::InvalidData => "InvalidData".to_string(),
            Error::PatchWelcome => "PatchWelcome".to_string(),
            Error::InputChanged => "InputChanged".to_string(),
            Error::OutputChanged => "OutputChanged".to_string(),
            Error::BsfNotFound => "BsfNotFound".to_string(),
            Error::DecoderNotFound => "DecoderNotFound".to_string(),
            Error::DemuxerNotFound => "DemuxerNotFound".to_string(),
            Error::EncoderNotFound => "EncoderNotFound".to_string(),
            Error::OptionNotFound => "OptionNotFound".to_string(),
            Error::MuxerNotFound => "MuxerNotFound".to_string(),
            Error::FilterNotFound => "FilterNotFound".to_string(),
            Error::ProtocolNotFound => "ProtocolNotFound".to_string(),
            Error::StreamNotFound => "StreamNotFound".to_string(),
            Error::HttpBadRequest => "HttpBadRequest".to_string(),
            Error::HttpUnauthorized => "HttpUnauthorized".to_string(),
            Error::HttpForbidden => "HttpForbidden".to_string(),
            Error::HttpNotFound => "HttpNotFound".to_string(),
            Error::HttpOther4xx => "HttpOther4xx".to_string(),
            Error::HttpServerError => "HttpServerError".to_string(),
        };

        f.write_str(&format!(
            "EffempegError({}:{})",
            description,
            self.as_raw_error()
        ))?;

        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self, f)?;
        Ok(())
    }
}

impl Error {
    pub fn as_raw_error(&self) -> c_int {
        match self {
            Error::Io(err) => err.raw_os_error().unwrap(),
            Error::Again => AVERROR_EAGAIN,
            Error::BsfNotFound => AVERROR_BSF_NOT_FOUND,
            Error::Bug => AVERROR_BUG,
            Error::BufferTooSmall => AVERROR_BUFFER_TOO_SMALL,
            Error::DecoderNotFound => AVERROR_DECODER_NOT_FOUND,
            Error::DemuxerNotFound => AVERROR_DEMUXER_NOT_FOUND,
            Error::EncoderNotFound => AVERROR_ENCODER_NOT_FOUND,
            Error::Eof => AVERROR_EOF,
            Error::Exit => AVERROR_EXIT,
            Error::External => AVERROR_EXTERNAL,
            Error::FilterNotFound => AVERROR_FILTER_NOT_FOUND,
            Error::InvalidData => AVERROR_INVALIDDATA,
            Error::MuxerNotFound => AVERROR_MUXER_NOT_FOUND,
            Error::OptionNotFound => AVERROR_OPTION_NOT_FOUND,
            Error::PatchWelcome => AVERROR_PATCHWELCOME,
            Error::ProtocolNotFound => AVERROR_PROTOCOL_NOT_FOUND,
            Error::StreamNotFound => AVERROR_STREAM_NOT_FOUND,
            Error::Bug2 => AVERROR_BUG2,
            Error::Unknown => AVERROR_UNKNOWN,
            Error::Experimental => AVERROR_EXPERIMENTAL,
            Error::InputChanged => AVERROR_INPUT_CHANGED,
            Error::OutputChanged => AVERROR_OUTPUT_CHANGED,
            Error::HttpBadRequest => AVERROR_HTTP_BAD_REQUEST,
            Error::HttpUnauthorized => AVERROR_HTTP_UNAUTHORIZED,
            Error::HttpForbidden => AVERROR_HTTP_FORBIDDEN,
            Error::HttpNotFound => AVERROR_HTTP_NOT_FOUND,
            Error::HttpOther4xx => AVERROR_HTTP_OTHER_4XX,
            Error::HttpServerError => AVERROR_HTTP_SERVER_ERROR,
        }
    }
}

// XXX: the length has to be synced with the number of errors
static mut STRINGS: [[c_char; AV_ERROR_MAX_STRING_SIZE as usize]; 27] =
    [[0; AV_ERROR_MAX_STRING_SIZE as usize]; 27];

impl error::Error for Error {}

impl Clone for Error {
    fn clone(&self) -> Error {
        self.as_raw_error().into()
    }
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.as_raw_error() == other.as_raw_error()
    }
}

impl Eq for Error {}

impl Hash for Error {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_raw_error().hash(state);
    }
}
