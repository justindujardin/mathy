build : mathy/*.py setup.py .env
	rm -f dist/*.whl
	.env/bin/python setup.py bdist_wheel

.env: setup.py requirements.txt
	virtualenv -p python3.6 .env
	.env/bin/pip install wheel
	.env/bin/pip install -r requirements.txt

clean:
	rm -rf dist
	rm -rf build
	rm -rf .env
	rm -rf mathy.egg-info
