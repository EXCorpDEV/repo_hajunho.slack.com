//Copyright (c) 2024 mynameis@hajunho.com. All rights reserved.


#include "RESTAPI_TesterBPLibrary.h"
#include "RESTAPI_Tester.h"
#include "JsonObjectConverter.h"

URESTAPI_TesterBPLibrary::URESTAPI_TesterBPLibrary(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{

}

float URESTAPI_TesterBPLibrary::RESTAPI_TesterSampleFunction(float Param)
{
    // debug msg
    UE_LOG(LogTemp, Warning, TEXT("RESTAPI_TesterSampleFunction called with Param: %f"), Param);
    
    // call restapi to unrealjunho's test server
    FString Url = "http://61.32.96.242:8000/howtotest/jjj";
    FString Verb = "GET";
    MakeRestApiCall(Url, Verb);

    return Param;
}

void URESTAPI_TesterBPLibrary::MakeRestApiCall(const FString& Url, const FString& Verb)
{
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->OnProcessRequestComplete().BindStatic(&URESTAPI_TesterBPLibrary::OnResponseReceived);
    Request->SetURL(Url);
    Request->SetVerb(Verb);
    Request->SetHeader(TEXT("User-Agent"), TEXT("X-UnrealEngine-Agent"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->ProcessRequest();
    
    UE_LOG(LogTemp, Display, TEXT("MakeRestApiCall : Sending HTTP request to: %s"), *Url);
}

void URESTAPI_TesterBPLibrary::OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful)
{
    if (bWasSuccessful && Response.IsValid())
    {
        FString ResponseString = Response->GetContentAsString();
        
        UE_LOG(LogTemp, Display, TEXT("RESTAPI_Tester : Received HTTP Response:"));
        UE_LOG(LogTemp, Display, TEXT("RESTAPI_Tester : Status Code: %d"), Response->GetResponseCode());
        UE_LOG(LogTemp, Display, TEXT("RESTAPI_Tester : Content-Type: %s"), *Response->GetContentType());
        UE_LOG(LogTemp, Display, TEXT("RESTAPI_Tester : Content:"));
        UE_LOG(LogTemp, Display, TEXT("%s"), *ResponseString);
        
        TSharedPtr<FJsonObject> JsonObject;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseString);
        
        if (FJsonSerializer::Deserialize(Reader, JsonObject))
        {
            UE_LOG(LogTemp, Display, TEXT("RESTAPI_Tester : Successfully parsed JSON response"));
            
            for (auto& Elem : JsonObject->Values)
            {
                UE_LOG(LogTemp, Display, TEXT("RESTAPI_Tester : JSON Field - %s: %s"),
                    *Elem.Key,
                    *Elem.Value->AsString());
            }
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("RESTAPI_Tester : Response is not in JSON format or failed to parse"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("RESTAPI_Tester : HTTP Request failed"));
        if (Response.IsValid())
        {
            UE_LOG(LogTemp, Error, TEXT("RESTAPI_Tester : Response Code: %d"), Response->GetResponseCode());
        }
    }
}
