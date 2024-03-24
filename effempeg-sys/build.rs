use std::{io::Cursor, path::PathBuf, process::Command};

fn get_configuration() -> String {
    if cfg!(feature = "standard") {
        "standard".to_string()
    } else {
        println!("falling back to standard mode");
        "standard".to_string()
    }
}

fn output() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
}

fn get_target() -> String {
    let output = Command::new("rustc").arg("-vV").output().unwrap();
    let output = std::str::from_utf8(&output.stdout).unwrap();

    let field = "host: ";
    let host = output
        .lines()
        .find(|l| l.starts_with(field))
        .map(|l| &l[field.len()..])
        .unwrap()
        .to_string();
    host
}

fn get_shared_libs_url() -> String {
    let configuration = get_configuration();
    let target = get_target();
    format!(
        "https://github.com/federico-terzi/effempeg/releases/download/build/{configuration}_sharedlibs_{target}.zip",
    )
}

fn get_bindings_url() -> String {
    let configuration = get_configuration();
    let target = get_target();
    format!(
        "https://github.com/federico-terzi/effempeg/releases/download/build/{configuration}_bindings_{target}.rs"
    )
}

fn get_shared_libs_folder() -> PathBuf {
    output().join("sharedlibs")
}

fn download_shared_libs() {
    let output_path = output().join("sharedlibs.zip");
    if !output_path.exists() {
        println!("downloading sharedlibs...");
        reqwest::blocking::get(get_shared_libs_url())
            .unwrap()
            .copy_to(&mut std::fs::File::create(&output_path).unwrap())
            .unwrap();
    }

    if !get_shared_libs_folder().exists() {
        let archive = std::fs::read(&output_path).unwrap();
        zip_extract::extract(Cursor::new(&archive), &get_shared_libs_folder(), true).unwrap();
    }
}

fn download_bindings() {
    let output_path = output().join("bindings.rs");
    if !output_path.exists() {
        println!("downloading bindings...");
        reqwest::blocking::get(get_bindings_url())
            .unwrap()
            .copy_to(&mut std::fs::File::create(&output_path).unwrap())
            .unwrap();
    }
}

fn download_resources() {
    download_shared_libs();
    download_bindings();
}

#[cfg(target_os = "macos")]
fn link_libraries() {
    let shared_libs_folder = get_shared_libs_folder().join("SHARED_LIBS");
    for entry in std::fs::read_dir(&shared_libs_folder).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        // On macOS, we generate multiple instances of dinamic libraries (not sure why)
        // This line allows us to link only to the main one.
        if path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .chars()
            .any(char::is_numeric)
        {
            continue;
        }

        let mut libname = path.file_stem().unwrap().to_str().unwrap();
        if libname.starts_with("lib") {
            libname = &libname[3..];
        }
        // println!("cargo:warning={:?}", libname);

        println!("cargo:rustc-link-lib=dylib={}", libname);
        println!(
            "cargo:rustc-link-search=native={}",
            shared_libs_folder.display()
        );
    }
}

#[cfg(target_os = "windows")]
fn link_libraries() {
    let shared_libs_folder = get_shared_libs_folder();
    for entry in std::fs::read_dir(&shared_libs_folder).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if !path.extension().unwrap().to_str().unwrap().eq("lib") {
            continue;
        }

        if path.is_file() {
            println!(
                "cargo:rustc-link-lib=dylib={}",
                path.file_stem().unwrap().to_str().unwrap()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                shared_libs_folder.display()
            );
        }
    }
}

fn main() {
    download_resources();
    link_libraries();
}
