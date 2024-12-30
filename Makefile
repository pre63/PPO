
default: sb3


sb3:
	@PYTHONPATH=. python sb3.py | tee -a .log/sb3.log


install:
	@sudo apt-get -y install swig || brew install swig
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p .log


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

