from fastapi import FastAPI
import pickle





app = FastAPI()
feature_extraction = pickle.load(open("feature_extraction.sav","rb"))
model = pickle.load(open("spamMail.sav","rb"))


@app.post("/v1")
def preDict(input_mail:str):
    input_mail = [input_mail]
    input_mail = feature_extraction.transform(input_mail)
    result = model.predict(input_mail)
    if result[0] == 1:
        return {"result":"It is a genuine mail [Ham]."}
    else:
        return {"result":"It is a fake/spam mail."}
 
        

