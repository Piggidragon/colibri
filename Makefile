.PHONY: all deepseek-v4 test check clean install uninstall

all deepseek-v4 test check clean install uninstall:
	$(MAKE) -C c $@