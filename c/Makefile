# DeepSeek V4 Flash is the sole supported runtime in this Linux x86-64 fork.
# Keep the public targets here; the amalgamation's object-unit graph remains in
# Makefile.deepseek-v4, where each target unit can be built independently.

PYTHON ?= python3
ARCH ?= native
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
LIBEXECDIR ?= $(PREFIX)/libexec/colibri
INSTALL ?= install

.PHONY: all deepseek-v4 deepseek-v4-clean deepseek-v4-tiny-generate \
	deepseek-v4-tiny-check deepseek-v4-oracle test check clean install uninstall

all: deepseek-v4

deepseek-v4:
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4

deepseek-v4-clean:
	$(MAKE) -f Makefile.deepseek-v4 deepseek-v4-clean

deepseek-v4-tiny-generate:
	$(PYTHON) tools/make_deepseek_v4_tiny.py --output ./deepseek_v4_tiny --force

deepseek-v4-tiny-check: deepseek-v4-tiny-generate
	$(MAKE) -f Makefile.deepseek-v4 deepseek-v4-clean
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4
	$(PYTHON) tests/test_deepseek_v4_tiny.py --binary ./deepseek_v4 --fixture ./deepseek_v4_tiny
	$(PYTHON) tests/test_deepseek_v4_prefix.py --binary $(CURDIR)/deepseek_v4 --fixture $(CURDIR)/deepseek_v4_tiny

deepseek-v4-oracle: deepseek-v4
	@test -n "$(MODEL)" || { echo "usage: make deepseek-v4-oracle MODEL=/path/to/checkpoint" >&2; exit 2; }
	$(PYTHON) tools/make_deepseek_v4_oracle.py --model "$(MODEL)" --binary ./deepseek_v4 --output tests/deepseek_v4_oracle.json --validate

test:
	$(PYTHON) -m unittest \
		tests.test_v4_dependency_closure \
		tests.test_v4_attention_source \
		tests.test_deepseek_v4_dspark_source \
		tests.test_v4_cli

check: deepseek-v4 test

install: deepseek-v4
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBEXECDIR) $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 coli $(DESTDIR)$(BINDIR)/coli
	$(INSTALL) -m 755 deepseek_v4 $(DESTDIR)$(LIBEXECDIR)/deepseek_v4
	$(INSTALL) -m 644 resource_plan.py doctor.py autotune.py openai_server.py version.py $(DESTDIR)$(LIBEXECDIR)/
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/coli $(DESTDIR)$(LIBEXECDIR)/deepseek_v4

clean: deepseek-v4-clean
	$(PYTHON) tools/clean.py
