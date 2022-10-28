Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install jianyao_forse

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``after_training12amin.post_training`` function:

.. autofunction:: after_training12amin.post_training

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`after_training12amin.post_training`
will raise an exception.

.. autoexception:: after_training12amin.post_training

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

