Releasing
=========

.. code-block:: sh

    # Bump version numbers
    git ls-files | xargs grep --color 2019\.1
    edit ...
    git commit -am"Bump version numbers to 2019.2.0"

    # Wait for CircleCI to report green and fix problems

    # Create tag and push
    git tag -m"Release version 2019.2.0" 2019.2.0
    git push 2019.2.0

    # Package and push to PyPI
    git clean -fdx
    python3 setup.py sdist
    python3 setup.py bdist_wheel --universal
    pip3 install --upgrade --user twine
    twine upload dist/*

    # Bump version numbers
    edit ...
    git commit -am"Bump version numbers to 2019.3.0.dev0"
