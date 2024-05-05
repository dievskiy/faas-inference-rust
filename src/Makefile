include .env
export

DOCKER_IMAGE_CONTAINER := "docker.io/dievskiy/inference:3"
DOCKER_IMAGE_UNIKERNEL := "ghcr.io/dievskiy/inference-unikernel:512mb"
DOCKER_IMAGE_WASM := "docker.io/dievskiy/inference-wasm:1"
DOCKER_IMAGE_WASM_NO_AOT := "docker.io/dievskiy/inference-wasm-no-aot:1"
UNIKERNEL_IMAGE_REGISTRY_CREDS := $(or $(UNIKERNEL_CREDS), "dievskiy:ghp_xxx")

DOCKER_IMAGE_CONTAINER_BUILT := .dockerimagebuilt

NATIVE_BUILD_DIR := native
NATIVE_TARGET_FILES = $(NATIVE_BUILD_DIR)/src/main.rs \
		      $(NATIVE_BUILD_DIR)/src/imagenet_classes.rs

WASM_BUILD_DIR := wasm
WASM_SOURCE_FILES = $(WASM_BUILD_DIR)/src/main.rs \
		    $(WASM_BUILD_DIR)/src/imagenet_classes.rs
WASM_TARGET_FILES = $(WASM_BUILD_DIR)/libtensorflowlite_c.so \
		    $(WASM_BUILD_DIR)/libtensorflowlite_flex.so \
		    $(WASM_BUILD_DIR)/libwasmedgePluginWasiNN.so

# MEMORY_LIMIT := 1073741824
MEMORY_LIMIT = 536870912
CPU_LIMIT := 1

all: native container microvm unikernel wasm

$(NATIVE_TARGET_FILES):
	touch $@

build_native: $(NATIVE_TARGET_FILES)
	# build statically
	cd $(NATIVE_BUILD_DIR) && RUSTFLAGS='-C target-feature=+crt-static' cargo build --target x86_64-unknown-linux-gnu --release

native_run:
	taskset -c 2 ./native/target/x86_64-unknown-linux-gnu/release/native

native: build_native
	echo "Native built"
	./native/target/x86_64-unknown-linux-gnu/release/native

$(DOCKER_IMAGE_CONTAINER_BUILT): build_native
	docker build -t $(DOCKER_IMAGE_CONTAINER) -f Dockerfile.container .
	docker push $(DOCKER_IMAGE_CONTAINER)
	touch $(DOCKER_IMAGE_CONTAINER_BUILT)

build_container: $(DOCKER_IMAGE_CONTAINER_BUILT)
	echo "Container built"

container_run:
	ctr run --cpus $(CPU_LIMIT) --memory-limit $(MEMORY_LIMIT) --rm $(DOCKER_IMAGE_CONTAINER) rust-inference-cont

gvisor_run:
	ctr run --cpus $(CPU_LIMIT) --runtime io.containerd.runsc.v1 --memory-limit $(MEMORY_LIMIT) --rm $(DOCKER_IMAGE_CONTAINER) rust-inference-gvisor

container: build_container
	echo "Sleeping for 3 seconds to make sure container was added"
	sleep 3
	ctr i pull $(DOCKER_IMAGE_CONTAINER)
	ctr run --cpus $(CPU_LIMIT) --memory-limit $(MEMORY_LIMIT) --rm $(DOCKER_IMAGE_CONTAINER) rust-inference-cont

$(WASM_TARGET_FILES):
	if [ ! -f $@ ]; then \
		find ~/.wasmedge -name $(notdir $@) | xargs cp -t ./wasm; \
	else \
        	echo "$@ already exists. Skipping."; \
    	fi

microvm_run:
	sudo firecracker-ctr --address /run/firecracker-containerd/containerd.sock run --snapshotter devmapper --runtime aws.firecracker --rm --memory-limit $(MEMORY_LIMIT) $(DOCKER_IMAGE_CONTAINER) microvm 2>&1

microvm: build_container
	echo "microvm built"
	sudo firecracker-ctr --address /run/firecracker-containerd/containerd.sock i pull $(DOCKER_IMAGE_CONTAINER)
	sudo firecracker-ctr --address /run/firecracker-containerd/containerd.sock run --snapshotter devmapper --runtime aws.firecracker --rm --memory-limit $(MEMORY_LIMIT) --net-host $(DOCKER_IMAGE_CONTAINER) microvm

build_unikernel: build_container
	cd unikernel && kraft build --plat fc --arch x86_64
	echo "Unikernel built"

unikernel_run:
	ctr run --cni --rm --runtime io.containerd.urunc.v2 --snapshotter devmapper $(DOCKER_IMAGE_UNIKERNEL) unikernel

unikernel: build_unikernel
	# make sure we can run the unikernel without container wrapper
	# cd unikernel && kraft run --plat fc --arch x86_64 --memory 1024M
	cd unikernel && bima build -t $(DOCKER_IMAGE_UNIKERNEL) -f Containerfile .
	ctr i push $(DOCKER_IMAGE_UNIKERNEL) -u $(UNIKERNEL_IMAGE_REGISTRY_CREDS)
	ctr run --cni --rm --runtime io.containerd.urunc.v2 --snapshotter devmapper $(DOCKER_IMAGE_UNIKERNEL) unikernel

# use it to make sure that kraftkit and firecracker are set up correctly
firecracker: build_unikernel
	touch /tmp/fireracker.log
	cd unikernel && firecracker --no-api --config-file fc.json

build_wasm: $(WASM_SOURCE_FILES)
	cd wasm && cargo build --release --target=wasm32-wasi

# optimize WASM module
compile_wasm: build_wasm
	wasmedge compile $(WASM_BUILD_DIR)/target/wasm32-wasi/release/wasm-inference.wasm $(WASM_BUILD_DIR)/inference.wasm
	echo "WASM built"

wasm_local: $(WASM_TARGET_FILES) compile_wasm
	wasmedge --dir .:. $(WASM_BUILD_DIR)/inference.wasm

wasm_local_no_aot: $(WASM_TARGET_FILES) compile_wasm
	wasmedge --dir .:. $(WASM_BUILD_DIR)/target/wasm32-wasi/release/wasm-inference.wasm

wasm_run:
	ctr run --cpus $(CPU_LIMIT) --memory-limit $(MEMORY_LIMIT) --rm --runc-binary crun --runtime io.containerd.runc.v2 --label module.wasm.image/variant=compat-smart $(DOCKER_IMAGE_WASM) wasm-example /inference.wasm

wasm_run_no_aot:
	ctr run --cpus $(CPU_LIMIT) --memory-limit $(MEMORY_LIMIT) --rm --runc-binary crun --runtime io.containerd.runc.v2 --label module.wasm.image/variant=compat-smart $(DOCKER_IMAGE_WASM_NO_AOT) wasm-example-no-aot /inference.wasm

wasm_no_aot: $(WASM_TARGET_FILES) compile_wasm
	buildah build --label "module.wasm.image/variant=compat-smart" -t $(DOCKER_IMAGE_WASM_NO_AOT) -f Dockerfile.wasm.no-aot .
	buildah push --authfile ~/.docker/config.json $(DOCKER_IMAGE_WASM_NO_AOT) docker://$(DOCKER_IMAGE_WASM_NO_AOT)
	ctr i pull $(DOCKER_IMAGE_WASM_NO_AOT)
	ctr run --cpus $(CPU_LIMIT) --memory-limit $(MEMORY_LIMIT) --rm --runc-binary crun --runtime io.containerd.runc.v2 --label module.wasm.image/variant=compat-smart $(DOCKER_IMAGE_WASM_NO_AOT) wasm-example /inference.wasm

wasm: $(WASM_TARGET_FILES) compile_wasm
	buildah build --label "module.wasm.image/variant=compat-smart" -t $(DOCKER_IMAGE_WASM) -f Dockerfile.wasm .
	buildah push --authfile ~/.docker/config.json $(DOCKER_IMAGE_WASM) docker://$(DOCKER_IMAGE_WASM)
	ctr i pull $(DOCKER_IMAGE_WASM)
	ctr run --cpus $(CPU_LIMIT) --memory-limit $(MEMORY_LIMIT) --rm --runc-binary crun --runtime io.containerd.runc.v2 --label module.wasm.image/variant=compat-smart $(DOCKER_IMAGE_WASM) wasm-example /inference.wasm
