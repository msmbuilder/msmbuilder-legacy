export PATH=$HOME/miniconda/bin:$PATH
export PATH=$HOME/miniconda/envs/$python/bin:$PATH

if [[ "$TRAVIS_PULL_REQUEST" == "true" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi

PYVERSION=$(python -c "import sys; print(sys.version_info[:2])")
if [[ "$PYVERSION" != "(2, 7)" ]]; then
    echo "No deploy on PYVERSION=${PYVERSION}"; exit 0
fi


if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi

# Create the docs and push them to S3
# sudo conda install --yes boto
pip install -q sphinx boto sphinx_rtd_theme bibpy
cd docs/sphinx && make html && cd -
python tools/ci/push-docs-to-s3.py
