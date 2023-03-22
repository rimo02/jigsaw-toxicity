import uvicorn
from fastapi import FastAPI,Request,Form,UploadFile,File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app import make_prediction

templates=Jinja2Templates(directory="templates")

app=FastAPI()
app.mount("/static",StaticFiles(directory="Static"),name="static")

@app.get('/',response_class=HTMLResponse)
def home(request:Request):
    context={'request':request}
    return templates.TemplateResponse('index.html',context)

@app.post('/upload',response_class=HTMLResponse)
async def get_preds(request:Request,message:str=Form(...)):
    pred=make_prediction(message)
    list1=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    out=dict(zip(list1,pred))
    result={}
    for key,value in out.items():
        if value==1:
            result[key]=value
    result=list(result.keys())
    context={'request':request,'msg':message,'output':result}
    return templates.TemplateResponse('index.html',context)


if __name__=="__main__":
    uvicorn.run(app)
