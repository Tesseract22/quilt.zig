const std = @import("std");
// const raylib = @cImport(@cInclude("raylib.h"));
const lodepng = @cImport(@cInclude("lodepng.h"));
const fftnd = @cImport(@cInclude("kiss_fftndr.h"));
const Vec4 = @Vector(4, f32);

pub fn Array(comptime T: type, comptime Depth: comptime_int) type {
    const IndexErr = error { OutOfBoundError };
    return struct {
        const Self = @This();

        shape: [Depth]usize,
        allocator: ?std.mem.Allocator,
        data: []T,
        fn assertDepth(comptime D: comptime_int, comptime methods: type) type {
            return if (D == Depth) methods else struct {}; 
        }
        // pub fn shuffle(self: Self, offsets: [Depth]isize) void {
        //     var is = [_]usize {0} ** Depth;
        //     for (0..Depth) |d| {
        //         while (is[d] < self.shape[d]): (is[d] += 1) {

        //         }
        //     }
        // }
        pub fn init(shape: [Depth]usize, allocator: std.mem.Allocator) !Self {
            var size: usize = 1;
            for (shape) |s| {
                size *= s;
            }
            return .{.shape = shape, .data = try allocator.alloc(T, size), .allocator = allocator};
        }
        pub fn deinit(self: Self) void {
            if (self.allocator) |a| a.free(self.data);
        }
        pub fn initWithRaw(shape: [Depth]usize, data: []T) !Self {
            var size: usize = 1;
            for (shape) |s| {
                size *= s;
            }
            std.debug.assert(data.len == size);
            return .{.shape = shape, .data = data, .allocator = null};
        }
        pub fn index(self: Self, pos: [Depth]usize) usize {
            var mul: usize = 1;
            var offset: usize = 0;
            for (0..Depth) |i| {
                offset += pos[Depth - i - 1] * mul;
                mul *= self.shape[Depth - i - 1];
            }
            return offset;
        }
        pub fn unIndex(self: Self, i: usize) [Depth]usize {
            var id = i;
            var pos: [Depth]usize = undefined;
            for (0..Depth) |d| {
                pos[Depth - d - 1] = id % self.shape[Depth - d - 1];
                id /= self.shape[Depth - d - 1];
            }
            return pos;
        }
        pub fn at(self: Self, pos: [Depth]usize) IndexErr!T {
            // x * height * depth + y * depth + z
            if (!self.checkBound(pos)) return IndexErr.OutOfBoundError;
            return self.atUnchecked(pos);
        }
        pub inline fn atUnchecked(self: Self, pos: [Depth]usize) T {
            return self.data[self.index(pos)];
        }
        pub fn get(self: Self, pos: [Depth]usize) IndexErr!*T {
            if (!self.checkBound(pos)) return IndexErr.OutOfBoundError;
            return self.getUnchecked(pos);
        }
        pub inline fn getUnchecked(self: Self, pos: [Depth]usize) *T {
            return &self.data[self.index(pos)];
        }
        pub fn set(self: Self, pos: [Depth]usize, val: T) IndexErr!void {
            // x * height * depth + y * depth + z
            if (!self.checkBound(pos)) return IndexErr.OutOfBoundError;
            return self.setUnchecked(pos, val);
        }
        pub inline fn setUnchecked(self: Self, pos: [Depth]usize, val: T) void {
            self.data[self.index(pos)] = val;
        }
        pub inline fn checkBoundForAxis(self: Self, comptime axis: comptime_int, i: usize) bool {
            if (axis >= Depth) @compileError("mismatch index of axises");
            return i >= 0 and i < self.shape[axis];
        }
        pub inline fn checkBound(self: Self, pos: [Depth]usize) bool {
            inline for (0..Depth) |axis| {
                if (!self.checkBoundForAxis(axis, pos[axis])) return false;
            }
            return true;
        }
        const CloneErr = error { NoAllocator, InvalidSlice };
        pub fn clone(self: Self, allocator: std.mem.Allocator) !Self {
            const res = try Self.init(self.shape, allocator);
            @memcpy(res.data, self.data);
            return res;

        }
        // pub fn cloneSlice(self: Self, start: [Depth]usize, end: [Depth]usize, allocator: std.mem.Allocator) !Self {
        //     var shape: [Depth]usize = undefined;
        //     for (0..Depth) |d| {
        //         if (end[d] <= start[d] or end[d] > self.shape[d]) return CloneErr.InvalidSlice;
        //         shape[d] = end[d] - start[d];
        //     }
        //     const res = try Self.init(shape, allocator);
        //     var pos_self = start;
        //     var pos_res = [_]usize {0} ** Depth;
        //     std.debug.print("{any} {any}\n", .{start, end});
        //     for (0..Depth) |d| {
        //         while (pos_self[d] < end[d]) {
        //             std.debug.print("{any} {any}\n", .{pos_res, pos_self, res.shape});
        //             res.setUnchecked(pos_res, self.atUnchecked(pos_self));

        //             pos_res[d] += 1;
        //             pos_self[d] += 1;
        //         }
        //     }
        //     return res;
        // }
        const info = @typeInfo(T);
        pub usingnamespace switch (info) {
            // std.builtin.Type
            .Vector,
            .Int,
            .Float
            => ArithmeticMethods(T, Depth),
            else => struct {},
        };
        pub usingnamespace switch (T) {
            Vec4 => assertDepth(2, ImageF32Methods),
            u8 => assertDepth(3, ImageU8Methods),
            f32 => assertDepth(2, FMatMethods),
            else => struct {}
        };
    };
}
pub const ImageF32 = Array(@Vector(4, f32), 2);
pub const ImageU8 = Array(u8, 3);
pub const CpxArr = Array(fftnd.kiss_fft_cpx, 2);
pub const F = fftnd.kiss_fft_scalar;
pub const FMat = Array(F, 2);
pub fn ArithmeticMethods(comptime T: type, comptime Depth: comptime_int) type {
    return struct {
        const Arr = Array(T, Depth);
        pub fn mul(a: Arr, b: Arr, allocator: std.mem.Allocator) Arr {
            for (0..Depth) |d| {
                std.debug.assert(a.shape[d] == b.shape[d]);
            }
            const res = a.clone(allocator) catch unreachable;
            for (a.data, b.data, res.data) |ai, bi, *ri| {
                ri.* = ai * bi;
            }
            return res;
        }
        pub fn pow(a: Arr, power: T, allocator: std.mem.Allocator) Arr {
            const res = a.clone(allocator) catch unreachable;
            for (a.data, res.data) |ai, *ri| {
                ri.* = std.math.pow(T, ai, power);
            }
        }
    };
}
pub const BorderType = union(enum) {
    Zero,
    Reflect,
    Cyclic,
    Shift: [2]isize,
};
pub const ImageF32Methods = struct {
    pub fn getPaddedShape(im_shape: [2]usize, kernal_shape: [2]usize) [2]usize {
        const pwlog: f32 = @log2(@as(f32, @floatFromInt(im_shape[1] + kernal_shape[1] - 1)));
        const phlog: f32 = @log2(@as(f32, @floatFromInt(im_shape[0] + kernal_shape[0] - 1)));
        const pw: usize = @as(usize, 1) << @as(u6, @intFromFloat(@ceil(pwlog)));
        const ph: usize = @as(usize, 1) << @as(u6, @intFromFloat(@ceil(phlog)));
        return .{ph, pw};
    }
    pub fn splitRGBAToShape(src: ImageF32, target_shape: [2]usize, border: BorderType, allocator: std.mem.Allocator) [4]FMat {
        var res: [4]FMat = undefined;
        

        const ph, const pw = target_shape;
        for (0..4) |i| {
            res[i] = FMat.init(.{ph, pw}, allocator) catch unreachable;
            @memset(res[i].data, 0);
        }
        switch (border) {
            .Zero => {
                for (0..src.shape[0]) |y| {
                    for (0..src.shape[1]) |x| {
                        for (0..4) |c| {
                            res[c].setUnchecked(.{y, x}, src.atUnchecked(.{y, x})[c]);
                        }
                    }
                }
            },
            .Shift => |s| {
                for (0..src.shape[0]) |y| {
                    for (0..src.shape[1]) |x| {
                        for (0..4) |c| {
                            const yi: isize = @intCast(y);
                            const xi: isize = @intCast(x);
                            res[c].setUnchecked(
                                .{
                                    @intCast(@mod(yi + s[0], @as(isize, @intCast(target_shape[0])))), 
                                    @intCast(@mod(xi + s[1], @as(isize, @intCast(target_shape[1]))))}, 
                                src.atUnchecked(.{y, x})[c]);
                        }
                    }
                }
            },
            .Reflect => {
                // |54321|012345|43210|12345|43210|12345
                const ym: isize = @intCast(src.shape[0]-1);
                const xm: isize = @intCast(src.shape[1]-1);
                for (0..target_shape[0]) |y| {
                    for (0..target_shape[1]) |x| {
                        for (0..4) |c| {
                            const yi: isize = @as(isize, @intCast(y)) - @as(isize, @intCast((target_shape[0] - src.shape[0])/2));
                            const xi: isize = @as(isize, @intCast(x)) - @as(isize, @intCast((target_shape[1] - src.shape[1])/2));
                            const yr = @abs(
                                @mod(yi - ym, 2 * ym) - ym);
                            const xr =  @abs(
                                    @mod(xi - xm, 2 * xm) - xm);
                            res[c].setUnchecked(.{y, x}, src.atUnchecked(.{yr, xr})[c]);
                        }
                        // 4 =  10 - 6 - 0
                        // 3 = 10 - 6 - 1
                        // ...
                        // 0 = 10 - 6 - 4
                        // -1 = 10 - 6 - 5
                        // 5 = abs -5 = 10 - 6 - 9
                        // 4 = 10 - 6 - 10

                        
                    }
                }
            },
            .Cyclic => unreachable,
        }

        return res;
    }
    pub fn splitRGBA(src: ImageF32, allocator: std.mem.Allocator) [4]FMat {
        var res: [4]FMat = undefined;
        for (0..4) |i| {
            res[i] = FMat.init(src.shape, allocator) catch unreachable;

        }
        for (0..src.shape[0]) |y| {
            for (0..src.shape[1]) |x| {
                for (0..4) |c| {
                    res[c].setUnchecked(.{y, x}, src.atUnchecked(.{y, x})[c]);
                }
            }
        }
        return res;
    }
    pub fn convult(src: ImageF32, kernal: FMat, border: BorderType, allocator: std.mem.Allocator) ImageF32 {
        const image_res = ImageF32.init(src.shape, allocator) catch unreachable;
        const h, const w = src.shape;
        // rbga channels for fft and ifft
        const padded_shape = getPaddedShape(src.shape, kernal.shape);
        var image_channels  = splitRGBAToShape(src, padded_shape, 
            border, allocator);
        var image_fft_channels: [4]CpxArr = undefined;
        var image_ifft_channels: [4]FMat = undefined;
        const padded_shape_cint = [2]c_int { 
            @intCast(padded_shape[0]), @intCast(padded_shape[1])
        };
        for (0..4) |c| {
            image_fft_channels[c] = CpxArr.init(image_channels[0].shape, allocator) catch unreachable;
            image_ifft_channels[c] = FMat.init(image_channels[0].shape, allocator) catch unreachable;
        }
        defer for (0..4) |c| {
            image_channels[c].deinit();
            image_fft_channels[c].deinit();
            image_ifft_channels[c].deinit();
        };
        // kernal fft
        const padded_kernal = kernal.padKernal(image_channels[0].shape, allocator);
        defer padded_kernal.deinit();
        const kernal_fft = CpxArr.init(image_channels[0].shape, allocator) catch unreachable;
        const cfg_kernal = fftnd.kiss_fftndr_alloc(@ptrCast(&padded_shape_cint), 2, 0, null, null).?;
        defer fftnd.free(cfg_kernal);
        defer kernal_fft.deinit();
        fftnd.kiss_fftndr(cfg_kernal, padded_kernal.data.ptr, kernal_fft.data.ptr);
        // image fft
        for (0..4) |c| {
            const cfg = fftnd.kiss_fftndr_alloc(@ptrCast(&padded_shape_cint), 2, 0, null, null).?;
            defer fftnd.free(cfg);
            fftnd.kiss_fftndr(cfg, image_channels[c].data.ptr, image_fft_channels[c].data.ptr);
        }
        // convultion
        for (0..4) |c| {
            for (kernal_fft.data, image_fft_channels[c].data) |k, *i| {
                i.* = cpxMul(i.*, k);
            }
        }
        // ifft
        for (0..4) |c| {
            const cfg = fftnd.kiss_fftndr_alloc(@ptrCast(&padded_shape_cint), 2, 1, null, null).?;
            defer fftnd.free(cfg);
            fftnd.kiss_fftndri(cfg, image_fft_channels[c].data.ptr, image_ifft_channels[c].data.ptr);
        }
        const N: f32 = @floatFromInt(padded_shape_cint[0] * padded_shape_cint[1]);
        // std.debug.print("raw:\n", .{});

        // for (0..padded_shape[0]) |y| {
        //     for (0..padded_shape[1]) |x| {
        //         std.debug.print("{}, ", .{
        //             @as(usize, @intFromFloat(image_ifft_channels[0].atUnchecked(.{y, x})/N))});
        //     }
        //     std.debug.print("\n", .{});
        // }
        for (0..h) |y| {
            for (0..w) |x| {
                const vec = image_res.getUnchecked(.{y, x});
                for (0..4) |c| {
                    const val = switch (border) {
                        .Reflect => image_ifft_channels[c].atUnchecked(
                            .{y + (padded_shape[0] - src.shape[0])/2, x + (padded_shape[1] - src.shape[1])/2}),
                        else => image_ifft_channels[c].atUnchecked(.{y, x}),
                    } / N;
                    vec[c] = val;
                }
            }
        }
        return image_res;
    }
    pub fn convultPadded(channel: FMat, kernal: FMat, allocator: std.mem.Allocator) FMat {
       std.debug.assert(channel.shape[0] == kernal.shape[0] and channel.shape[1] == kernal.shape[1]);
        const res = FMat.init(channel.shape, allocator) catch unreachable;
        const h, const w = channel.shape;
        // rbga channels for fft and ifft
        var image_fft_channel = CpxArr.init(res.shape, allocator) catch unreachable;
        var image_ifft_channel = FMat.init(res.shape, allocator) catch unreachable;
        const padded_shape = [2]c_int { 
            @intCast(image_fft_channel.shape[0]), @intCast(image_fft_channel.shape[1])
        };

        defer image_fft_channel.deinit();
        defer image_ifft_channel.deinit();
        // kernal fft
        const kernal_fft = CpxArr.init(image_fft_channel.shape, allocator) catch unreachable;
        const cfg_kernal = fftnd.kiss_fftndr_alloc(@ptrCast(&padded_shape), 2, 0, null, null).?;
        defer fftnd.free(cfg_kernal);
        defer kernal_fft.deinit();
        fftnd.kiss_fftndr(cfg_kernal, kernal.data.ptr, kernal_fft.data.ptr);
        // image fft
        const cfg_foward = fftnd.kiss_fftndr_alloc(@ptrCast(&padded_shape), 2, 0, null, null).?;
        defer fftnd.free(cfg_foward);
        fftnd.kiss_fftndr(cfg_foward, channel.data.ptr, image_fft_channel.data.ptr);
        // convultion
        for (kernal_fft.data, image_fft_channel.data) |k, *i| {
            i.* = cpxMul(i.*, k);
        }
        // ifft
        const cfg_backward = fftnd.kiss_fftndr_alloc(@ptrCast(&padded_shape), 2, 1, null, null).?;
        defer fftnd.free(cfg_backward);
        fftnd.kiss_fftndri(cfg_backward, image_fft_channel.data.ptr, image_ifft_channel.data.ptr);
        for (0..h) |y| {
            for (0..w) |x| {
                const N: f32 = @floatFromInt(padded_shape[0] * padded_shape[1]);
                const val = image_ifft_channel.atUnchecked(.{y, x}) / N;
                res.setUnchecked(.{y, x}, val);
            }
        }
        return res;
    }
    pub fn convultImage(src: ImageF32, kernal: ImageF32, border: BorderType, allocator: std.mem.Allocator) ImageF32 {
        const padded_shape = getPaddedShape(src.shape, kernal.shape);
        const image_channels = splitRGBAToShape(src, padded_shape, border, allocator);
        const kernal_channels = splitRGBAToShape(kernal, padded_shape, 
            BorderType { .Shift = 
                .{-@as(isize, @intCast(kernal.shape[0]/2)), -@as(isize, @intCast(kernal.shape[1]/2)) }}, 
            allocator);
        defer for (0..4) |c| {
            image_channels[c].deinit();
            kernal_channels[c].deinit();
        };
        const res = ImageF32.init(src.shape, allocator) catch unreachable;
        for (0..4) |c| { // TODO: dont convult on alpha
            const convulted = convultPadded(image_channels[c], kernal_channels[c], allocator);
            defer convulted.deinit();
            for (0..res.shape[0]) |y| {
                for (0..res.shape[1]) |x| {
                    const val = switch (border) {
                        .Reflect => convulted.atUnchecked(
                            .{y + (padded_shape[0] - src.shape[0])/2, x + (padded_shape[1] - src.shape[1])/2}),
                        else => convulted.atUnchecked(.{y, x}),
                    };
                    res.getUnchecked(.{y, x})[c] = val;

                }
            }
        }
        return res;
    }
    pub fn ImageF32ToU8(src: ImageF32, allocator: std.mem.Allocator) ImageU8 {
        const res = ImageU8.init(.{src.shape[0], src.shape[1], 4}, allocator) catch unreachable;
        for (0..src.shape[0]) |y| {
            for (0..src.shape[1]) |x| {
                for (0..4) |c| {
                    res.setUnchecked(.{y, x, c}, 
                        @intFromFloat(@floor(src.atUnchecked(.{y, x})[c] * 255.0 + 0.5)));
                }
            }
        }
        return res;
    }
    pub fn toGreyScale(src: ImageF32, allocator: std.mem.Allocator) ImageF32 {
        const res = ImageF32.init(src.shape, allocator) catch unreachable;
        for (res.data, src.data) |*r, s| {
            r.* = @splat((s[0] + s[1] + s[2]) / 3);
            r[3] = 1;
        }
        return res;
    }
};

pub const ImageU8Methods = struct {
    pub fn ImageU8ToF32(src: ImageU8, allocator: std.mem.Allocator) ImageF32 {
        const res = ImageF32.init(.{src.shape[0], src.shape[1]}, allocator) catch unreachable;
        for (0..src.shape[0]) |y| {
            for (0..src.shape[1]) |x| {
                const vec = res.getUnchecked(.{y, x});
                for (0..4) |c| {
                    vec[c] = @as(f32, @floatFromInt(src.atUnchecked(.{y, x, c}))) / 255.0;
                }

            }
        }
        return res;
    }

};

pub const FMatMethods = struct {
    pub fn padKernal(k: FMat, other_shape: [2]usize, allocator: std.mem.Allocator) FMat {
        var res = FMat.init(other_shape, allocator) catch unreachable;
        @memset(res.data, 0);
        const kh, const kw = k.shape;
        const hh = kh / 2;
        const hw = kw / 2;
        for (0..kh) |y| {
            for (0..kw) |x| {
                const y_wrap = if (y < hh) other_shape[0] - hh + y else y - hh;
                const x_wrap = if (x < hw) other_shape[1] - hw + x else x - hw;
                res.setUnchecked(.{y_wrap, x_wrap}, k.atUnchecked(.{y, x}));
            }
        }
        // for (0..other_shape[1]) |x| {
        //     std.debug.print("{} {}\n", .{x, res.atUnchecked(.{0, x})});
        // }
        return res;
    }
};

pub inline fn cpxMul(a: fftnd.kiss_fft_cpx, b: fftnd.kiss_fft_cpx) fftnd.kiss_fft_cpx {
    return .{.r = a.r * b.r - a.i * b.i, .i = a.r * b.i + b.r * a.i};
}



pub fn gaussian2D(size: usize, sd: F, allocator: std.mem.Allocator) FMat {
    const res = FMat.init(.{size, size}, allocator) catch unreachable;
    const half: F = @floatFromInt(size / 2);
    const sd2 = sd*sd;
    var sum: f32 = 0;
    for (0..size) |y| {
        for (0..size) |x| {
            const dy = @as(f32, @floatFromInt(y)) - half;
            const dx = @as(f32, @floatFromInt(x)) - half;
            const d2 = dy*dy + dx*dx;
            const g: f32 = @floatCast(@exp(-d2/(2*sd*sd)) / (2*fftnd.M_PI*sd2));
            res.setUnchecked(.{y, x}, g);
            sum += g;
        }
    }
    for (res.data) |*i| {
        i.* /= sum;
    }
    return res;
}