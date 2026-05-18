try:
    import App.services.rating.rating
    print('OK')
except Exception as e:
    import traceback
    with open('err.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
