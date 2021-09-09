from fastai.learner import load_learner

def load_fastai_classifier(path: str):
    model_inf = load_learner(path)
    return model_inf

'''
classify function will take model and 
'''
def fastai_classify(model_inf, img=None):
    res1 = model_inf.predict(img)
    pred = res1[0]
    moulting_conf = res1[2][0].numpy()
    refl_conf = res1[2][1].numpy()
    return {"pred":pred, "moulting_conf":moulting_conf, "refl_conf":refl_conf}

