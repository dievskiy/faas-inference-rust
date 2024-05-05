This repository contains the code needed to build an ML inference application using DenseNet201 model and deploy it to container-, firecracker-, Wasm-, and unikraft-based runtimes.
The project requires x86_64 architecture.

## Native

Building native binary requires setting up x86_64-unknown-linux-gnu target as we build statically to reduce OCI image size for container, unikernel, and Wasm:

```
rustup add target x86_64-unknown-linux-gnu
make native
```

## Container

Simply use:

```
make container
```

## microVM

We use Firecracker as microVM platform. Make sure firecracker and firecracker-containerd are installed.
**Important**: microVM target requires nested virtualization features, so make sure your instance/VM has these abilities. You can use `lscpu | grep vmx` to verify that nested virtualization is supported on Intel chips.

Default firecracker-contaienrd uses extremely small amount of memory for VMs, so we need to patch it with microvm/firecracker-containerd.patch.

Next configure firecracker-containerd config.toml, vmlinux according to https://github.com/firecracker-microvm/firecracker-containerd/blob/main/docs/getting-started.md and devmapper device, and check that example app can be run:

```
sudo firecracker-ctr --address /run/firecracker-containerd/containerd.sock run --snapshotter devmapper --runtime aws.firecracker --rm --tty --net-host docker.io/library/busybox:latest busybox-test
# make sure firecracker process is running
ps aux | grep firecracker
```

Finally run our inference app using docker image built for container:

```
sudo firecracker-ctr --address /run/firecracker-containerd/containerd.sock run --snapshotter devmapper --runtime aws.firecracker --rm --tty --net-host docker.io/<user>/<image>:<tag> microvm
```

## Unikernel

We use Unikraft with app-elfloader and kraftkit for building unikernel. Refer to https://unikraft.org/guides/catalog-using-firecracker for trivial setup. The important thig is to install kraftkit, run buildkit container, and (optionally) set up tun/tap interface.

To create the container image (package unikernel in OCI format) we use [bima](https://github.com/nubificus/bima). To run the image we use [urunc](https://github.com/nubificus/urunc) unikernel runtime. urunc is patched to support elfloader and "com.urunc.unikernel.memory" Dockerfile label, so you need to build it from https://github.com/dievskiy/urunc/commits/unikraft-elfloader-integration/. You might also need to patch bima to incread timeout on image pulling as it can take more than 5 seconds in case the image is large (and that's true for images with tflite shared libraries).

Next configure .env according to .env.example to be able to push to registry with ctr and run:

```
make unikernel # might take 5-10 minutes
```

## Wasm

First, make sure that you have set up crun, wasmedge, and tensorflow wasi-nn plugin

1. crun (custom build to support wasi-nn, regular crun won't work)

```
git clone https://github.com/hydai/crun -b enable_plugin
./autogen.sh
./configure --with-wasmedge
make
make install
```

2. wasmedge

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-tensorflowlite
```

3. Configure rustup

```
rustup target add wasm32-wasi
```

4. Install buildah

5. Finally build Wasm module and run it with wasmedge container runtime:

```
make wasm
```
