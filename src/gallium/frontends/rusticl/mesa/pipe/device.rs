// Copyright 2020 Red Hat.
// SPDX-License-Identifier: MIT

use crate::pipe::screen::*;

use mesa_rust_gen::*;
use mesa_rust_util::ptr::ThreadSafeCPtr;
use mesa_rust_util::string::c_string_to_string;

use std::collections::HashMap;
use std::ffi::CStr;
use std::{env, ptr};

#[derive(PartialEq)]
pub(super) struct PipeLoaderDevice {
    ldev: ThreadSafeCPtr<pipe_loader_device>,
}

impl PipeLoaderDevice {
    fn new(ldev: *mut pipe_loader_device) -> Option<Self> {
        Some(Self {
            // SAFETY: `pipe_loader_device` is considered to be thread-safe
            ldev: unsafe { ThreadSafeCPtr::new(ldev)? },
        })
    }

    fn load_screen(self) -> Option<PipeScreenWithLdev> {
        let s = unsafe { pipe_loader_create_screen(self.ldev.as_ptr(), false) };
        PipeScreenWithLdev::new(self, s)
    }

    pub fn driver_name(&self) -> &CStr {
        // SAFETY: ldev is a valid memory allocation
        let ldev = unsafe { self.ldev.as_ref() };

        // SAFETY: The driver name is a valid C string pointer
        unsafe { CStr::from_ptr(ldev.driver_name) }
    }

    pub fn device_type(&self) -> pipe_loader_device_type {
        unsafe { self.ldev.as_ref().type_ }
    }
}

impl Drop for PipeLoaderDevice {
    fn drop(&mut self) {
        unsafe {
            pipe_loader_release(&mut self.ldev.as_ptr(), 1);
        }
    }
}

fn load_devs() -> impl Iterator<Item = PipeLoaderDevice> {
    let n = unsafe { pipe_loader_probe(ptr::null_mut(), 0, true) };
    let mut devices: Vec<*mut pipe_loader_device> = vec![ptr::null_mut(); n as usize];
    unsafe {
        pipe_loader_probe(devices.as_mut_ptr(), n, true);
    }

    devices.into_iter().filter_map(PipeLoaderDevice::new)
}

fn parse_enabled_devs(enabled_devs: &str) -> HashMap<String, u32> {
    let mut res = HashMap::new();

    // I wished we could use different iterators, but that's not really working out.
    let mut last_driver = None;
    for driver_str in enabled_devs.split(',') {
        if driver_str.is_empty() {
            continue;
        }

        // if the string parses to a number, just updated the device bitset
        if let Ok(dev_id) = driver_str.parse::<u8>() {
            if let Some(last_driver) = last_driver {
                *res.get_mut(last_driver).unwrap() |= 1 << dev_id;
            }
            continue;
        } else {
            let driver_str: Vec<_> = driver_str.split(':').collect();
            let mut devices = 0;

            if driver_str.len() == 1 {
                devices = !0;
            } else if let Ok(dev_id) = driver_str[1].parse::<u8>() {
                devices |= 1 << dev_id;
            }

            let driver_str = match driver_str[0] {
                "llvmpipe" | "lp" => "swrast",
                "freedreno" => "msm",
                a => a,
            };

            res.insert(driver_str.to_owned(), devices);
            last_driver = Some(driver_str);
        }
    }

    if res.contains_key("panfrost") {
        res.insert("panthor".to_owned(), res["panfrost"]);
    }

    if let Some(&mask) = res.get("asahi") {
        res.entry("aurora".to_owned()).or_insert(mask);
    } else if let Some(&mask) = res.get("aurora") {
        res.entry("asahi".to_owned()).or_insert(mask);
    }

    res
}

fn get_enabled_devs() -> HashMap<String, u32> {
    // we require the type here as this list can be empty depending on the build options
    let default_devs: &[&str] = &[
        #[cfg(any(rusticl_enable_asahi, rusticl_enable_auto))]
        "asahi",
        #[cfg(any(rusticl_enable_freedreno, rusticl_enable_auto))]
        "freedreno",
        #[cfg(any(rusticl_enable_panfrost, rusticl_enable_auto))]
        "panfrost",
        #[cfg(rusticl_enable_radeonsi)]
        "radeonsi",
    ];

    let enabled_devs = env::var("RUSTICL_ENABLE").unwrap_or(default_devs.join(","));
    parse_enabled_devs(&enabled_devs)
}

pub fn load_screens() -> impl Iterator<Item = PipeScreenWithLdev> {
    let devs = load_devs();
    let mut enabled_devs = get_enabled_devs();

    devs.filter(move |dev| {
        let driver_name = unsafe { c_string_to_string(dev.ldev.as_ref().driver_name) };

        if let Some(enabled_devs) = enabled_devs.get_mut(&driver_name) {
            let res = (*enabled_devs & 1) == 1;
            *enabled_devs >>= 1;

            res
        } else {
            false
        }
    })
    .filter_map(PipeLoaderDevice::load_screen)
}

#[cfg(test)]
mod tests {
    use super::parse_enabled_devs;

    #[test]
    fn asahi_and_aurora_alias_both_ways() {
        let from_asahi = parse_enabled_devs("asahi");
        assert_eq!(from_asahi.get("asahi"), Some(&!0));
        assert_eq!(from_asahi.get("aurora"), Some(&!0));

        let from_aurora = parse_enabled_devs("aurora");
        assert_eq!(from_aurora.get("aurora"), Some(&!0));
        assert_eq!(from_aurora.get("asahi"), Some(&!0));
    }

    #[test]
    fn asahi_and_aurora_preserve_device_masks() {
        let from_aurora_0 = parse_enabled_devs("aurora:0");
        assert_eq!(from_aurora_0.get("aurora"), Some(&1));
        assert_eq!(from_aurora_0.get("asahi"), Some(&1));

        let explicit_both = parse_enabled_devs("asahi:0,aurora:2");
        assert_eq!(explicit_both.get("asahi"), Some(&1));
        assert_eq!(explicit_both.get("aurora"), Some(&4));
    }

    #[test]
    fn unrelated_or_empty_enable_leaves_aurora_disabled() {
        assert!(!parse_enabled_devs("").contains_key("aurora"));
        let lp = parse_enabled_devs("llvmpipe");
        assert_eq!(lp.get("swrast"), Some(&!0));
        assert!(!lp.contains_key("asahi"));
        assert!(!lp.contains_key("aurora"));
    }
}
