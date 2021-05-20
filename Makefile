BASE_IMG_NAME = nam012
BUILD_CMD = docker build --network=host
DOCKERFILE = Dockerfile

USERNAME = $(shell whoami)
USERID = $(shell id -u)
GROUPID = $(shell id -g)

BUILD_ARGS = --build-arg user=$(USERNAME) \
             --build-arg uid=$(USERID) \
             --build-arg gid=$(GROUPID)

default:
	$(BUILD_CMD) $(BUILD_ARGS) -f $(DOCKERFILE) -t $(BASE_IMG_NAME)-iou .

clean: clean-th || clean-tf
	docker image rm $(BASE_IMG_NAME)-iou


