//Copyright (c) 2024 mynameis@hajunho.com. All rights reserved.


#include "RESTAPI_Tester.h"

#define LOCTEXT_NAMESPACE "FRESTAPI_TesterModule"

void FRESTAPI_TesterModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	UE_LOG(LogTemp, Warning, TEXT("restapiSampleFunction StartupModule called HJH"));
}

void FRESTAPI_TesterModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	UE_LOG(LogTemp, Warning, TEXT("restapiSampleFunction ShutdownModule called HJH"));
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FRESTAPI_TesterModule, RESTAPI_Tester)
