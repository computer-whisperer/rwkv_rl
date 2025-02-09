use std::ffi::c_void;



#[no_mangle]
pub extern "C" fn handleEvent(skirmish_ai_id: u32, topic: u32, data: *const c_void) -> u32{
    println!("Hello world!!!");
    0
}

