Releasing
=========

.. code-block:: sh

    # Bump version numbers
    git ls-files | xargs grep --color 2020\.1
    edit ...
    git commit -am"Bump version numbers to 2020.1.0"

    # Wait for CircleCI to report green and fix problems

    # Create tag and push
    git tag -m"Release version 2020.1.0" 2020.1.0
    git push 2020.1.0

    # Package and push to PyPI
    git clean -fdx
    python3 setup.py sdist
    python3 setup.py bdist_wheel --universal
    pip3 install --upgrade --user twine
    twine upload dist/*

    # Bump version numbers to 2021.1.0.dev0
    edit ...
