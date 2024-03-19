const std = @import("std");
const raysdk = @import("raylib/src/build.zig");
// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
fn dependencies(c: *std.Build.Step.Compile, raylib: @TypeOf(c)) void {
    // c.linkLibrary(raylib);
    _ = raylib;
    c.linkLibC();
    c.addIncludePath(.{.path="./raylib/src"});
    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    const loadepng_flags = &[_][]const u8{
        "-ansi",
        "-O3",
    };
    c.addIncludePath(.{ .path = "./lodepng" });
    c.addCSourceFile(.{ .file = .{ .path = "lodepng/lodepng.c" }, .flags = loadepng_flags });
    c.addIncludePath(.{.path = "./kissfft"});
    c.addCSourceFiles(
        .{
            .files = &.{
                "kissfft/kiss_fft.c", 
                "kissfft/kiss_fftnd.c", 
                "kissfft/kiss_fftndr.c", 
                "kissfft/kiss_fftr.c"}, 
            .flags = &.{
                "-O3",
                // "-DUSE_SIMD=1",
                // "-msse",
            }});
}
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});



    const exe = b.addExecutable(.{
        .name = "demo",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // const raylib = raysdk.addRaylib(b, target, optimize, .{}) catch unreachable;

    dependencies(exe, undefined);
    b.installArtifact(exe);

    const quilt_exe = b.addExecutable(.{
        .name = "quilt",
        .root_source_file = .{ .path = "src/quilt.zig" },
        .target = target,
        .optimize = optimize,
    });
    dependencies(quilt_exe, undefined);
    b.installArtifact(quilt_exe);
    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(quilt_exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);




    const exe_unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/quilt.zig" },
        .target = target,
        .optimize = optimize,
    });
    dependencies(exe_unit_tests, undefined);
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
