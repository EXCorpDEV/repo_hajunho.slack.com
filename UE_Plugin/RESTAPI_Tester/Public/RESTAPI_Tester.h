//Copyright (c) 2024 mynameis@hajunho.com. All rights reserved.

#pragma once

#include "Modules/ModuleManager.h"

class FRESTAPI_TesterModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
