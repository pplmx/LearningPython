.PHONY: help lock sync ruff init
.DEFAULT_GOAL := help

# define a func to check if the current OS is windows: if ${OS} is defined, it is windows
is_windows = $(if ${OS},true,false)

init:
	@# if .venv does not exist, create it
	@# if uv exists, use it to create the virtual environment
	@# otherwise, use python3
	@# install dependencies
	@if [ ! -d .venv ]; then \
		if [ -x "$(which uv)" ]; then \
			uv venv; \
		else \
			python3 -m venv .venv; \
		fi; \
	fi

# format and lint
ruff:
	@ruff format
	@ruff check --fix --unsafe-fixes --select I

# start jupyter lab
lab:
	@jupyter lab

# lock
lock:
	@# if pyproject.toml exists, use it to lock dependencies
	@if [ -f pyproject.toml ]; then \
		uv pip compile pyproject.toml -o requirements.txt; \
	else \
		uv pip freeze | uv pip compile - -o requirements.txt; \
	fi

# sync install from a requirements.txt file
sync:
	@uv pip sync requirements.txt

# Show help
help:
	@echo ""
	@echo "Usage:"
	@echo "    make [target]"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z\-_0-9]+:/ \
	{ \
		helpMessage = match(lastLine, /^# (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 2, RLENGTH); \
			printf "\033[36m%-22s\033[0m %s\n", helpCommand,helpMessage; \
		} \
	} { lastLine = $$0 }' $(MAKEFILE_LIST)
