FROM yolov5-base

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy repo contents
COPY . /usr/src/app

# Clean the weights and inference mountpoints, they must be mounted 
# at run-time
RUN rm -rf weights inference 
