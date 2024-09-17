/*
* This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <string>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace Util
{
	// Return value is whether the value was captured while being mandatory or
	// not
	template <typename T>
	bool getParam(json* j, T* buffer, const std::string& argname,
				  T defaultValue, bool isMandatory,
				  const std::string& errorMessage = "")
	{
		try
		{
			*buffer = j->at(argname);
			return true;
		}
		catch (json::out_of_range& e)
		{
			if (isMandatory)
			{
				if (errorMessage.empty())
				{
					throw std::invalid_argument("Missing mandatory argument " +
												argname + " in json file");
				}
				else
				{
					throw std::invalid_argument(errorMessage);
				}
			}
			*buffer = defaultValue;
			return false;
		}
	}

}