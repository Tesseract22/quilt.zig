const std = @import("std");
const fft = @import("convultion.zig");
// const raylib = @cImport(@cInclude("raylib.h"));
const lodepng = @cImport(@cInclude("lodepng.h"));

// fn guassianFilter()
pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();
    _ = stdout;
    var out: [*c]u8 = undefined;
    var w: c_uint = 0;
    var h: c_uint = 0;
    const err = lodepng.lodepng_decode32_file(@ptrCast(&out), @ptrCast(&w), @ptrCast(&h), "rock.png");
    defer std.c.free(out);
    // const image = out[0..w*h];
    std.log.debug("Error: {s}", .{lodepng.lodepng_error_text(err)});
    std.log.debug("w: {}, h: {}\n", .{w, h});
    const image = fft.ImageU8.initWithRaw(.{h, w, 4}, out[0..h*w*4]) catch unreachable;
    const image_f32 = image.ImageU8ToF32(allocator);
    defer image_f32.deinit();
    // kernal
    const kernal = fft.FMat.init(.{5,5}, allocator) catch unreachable;
    @memset(kernal.data, 0);
    kernal.setUnchecked(.{2,2}, 1);
    defer kernal.deinit();
    const convulted = image_f32.convult(kernal, .Zero, allocator);
    defer convulted.deinit();

    const image_res = convulted.ImageF32ToU8(allocator);
    defer image_res.deinit();

    for (0..10) |y| {
        for (0..10) |x| {
            for (0..4) |c| {
                if (image_res.atUnchecked(.{y, x, c}) != image.atUnchecked(.{y, x, c}))
                std.debug.print("[{}, {}, {}] = {}/{}\n", .{y, x, c, image_res.atUnchecked(.{y, x, c}), image.atUnchecked(.{y, x, c})});
                
            }
        }
    }
    const encode_err = lodepng.lodepng_encode32_file("output2.png", image_res.data.ptr, w, h);
    std.debug.print("{s}\n", .{lodepng.lodepng_error_text(encode_err)});
    
}









