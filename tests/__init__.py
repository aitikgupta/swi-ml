from swi_ml import set_automatic_fallback

# testing should run even for machines without 'cupy' backend
# enable falling back to 'numpy' for tests
set_automatic_fallback(True)
