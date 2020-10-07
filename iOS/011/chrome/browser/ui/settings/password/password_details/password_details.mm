// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/settings/password/password_details/password_details.h"

#include "base/strings/sys_string_conversions.h"
#include "components/autofill/core/common/password_form.h"
#include "components/password_manager/core/browser/android_affiliation/affiliation_utils.h"
#include "components/password_manager/core/browser/password_ui_utils.h"
#include "components/password_manager/core/browser/well_known_change_password_util.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

@implementation PasswordDetails

- (instancetype)initWithPasswordForm:(const autofill::PasswordForm&)form {
  self = [super init];
  if (self) {
    auto facetUri = password_manager::FacetURI::FromPotentiallyInvalidSpec(
        form.signon_realm);
    if (facetUri.IsValidAndroidFacetURI()) {
      if (!form.app_display_name.empty()) {
        _changePasswordURL = password_manager::CreateChangePasswordUrl(
            GURL(form.affiliated_web_realm));
        _origin = base::SysUTF8ToNSString(form.app_display_name);
        _website = base::SysUTF8ToNSString(form.app_display_name);
      } else {
        _origin = base::SysUTF8ToNSString(facetUri.android_package_name());
        _website = base::SysUTF8ToNSString(facetUri.android_package_name());
      }
    } else {
      auto nameWithLink = password_manager::GetShownOriginAndLinkUrl(form);
      _origin = base::SysUTF8ToNSString(nameWithLink.first);
      _website = base::SysUTF8ToNSString(nameWithLink.second.spec());
      _changePasswordURL = password_manager::CreateChangePasswordUrl(form.url);
    }
    _username = base::SysUTF16ToNSString(form.username_value);
    _password = base::SysUTF16ToNSString(form.password_value);
  }
  return self;
}

@end
