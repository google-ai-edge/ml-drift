# This file is used to build the cached Dawn binary locally

# 1. `cd third_party/ml_drift/release/`
# 2. Build the image locally
# `gcloud auth configure-docker us-docker.pkg.dev`
# `docker build . -t dawn -f kokoro/opt/dawn.Dockerfile`
# (Note that the docker build cmd will take ~10 minutes)
# 3. Run the the image
# `docker run -dit dawn`
# 4. Get the container id from running `docker ps`
# 5. Enter the docker container with `docker exec -it <container_id> bash`
# 6. Generate the tarball
# 'cd /tmp/dawn/'
# 'rm -rf webgpu-dawn-binaries/out/latest/_deps/'
# 'tar czf webgpu-dawn-binaries.tar.gz webgpu-dawn-binaries'
# 7. Exit the container with `exit`
# 8. Copy out the tarball with `docker cp <container_id>:/tmp/dawn/webgpu-dawn-binaries.tar.gz /tmp/`
# 9. Upload the tarball to /google/data/rw/teams/ml-drift/dawn/webgpu-dawn-binaries.tar.gz

FROM us-docker.pkg.dev/artifact-foundry-prod/docker-3p-trusted/ubuntu:22.04

COPY kokoro/common.sh .
RUN chmod +x ./common.sh
RUN ./common.sh

COPY dawn_scripts/build_libdawn.sh .
RUN chmod +x ./build_libdawn.sh
RUN ./build_libdawn.sh
