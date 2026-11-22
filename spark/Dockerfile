# Build only this directory: docker build --platform linux/arm64 -t ds4023-c-resnet18-spark:25.11 spark/
FROM nvcr.io/nvidia/pytorch:25.11-py3

COPY requirements-extra.txt validate_overlay.py /opt/traffic-sign-build/

# Capture the NVIDIA stack before creating a venv. No project data enters the image.
RUN python /opt/traffic-sign-build/validate_overlay.py snapshot \
      --output /opt/traffic-sign-protected.json \
    && python -m venv --without-pip --system-site-packages /opt/traffic-sign-venv

ENV VIRTUAL_ENV=/opt/traffic-sign-venv
ENV PATH="/opt/traffic-sign-venv/bin:${PATH}"
ENV MPLBACKEND=Agg

# Reuse the matched NVIDIA Torch/Torchvision stack. Missing application packages may be added
# inside the venv; NVIDIA's pip config and /etc/pip/constraint.txt remain in force.
RUN python /opt/traffic-sign-build/validate_overlay.py install \
      --baseline /opt/traffic-sign-protected.json \
      --requirements /opt/traffic-sign-build/requirements-extra.txt \
      --report /opt/traffic-sign-environment.json

WORKDIR /workspace
# Deliberately inherit NVIDIA's entrypoint and its runtime initialization.
