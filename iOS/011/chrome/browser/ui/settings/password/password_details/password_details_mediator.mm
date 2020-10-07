// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/settings/password/password_details/password_details_mediator.h"

#include "base/strings/sys_string_conversions.h"
#include "components/autofill/core/common/password_form.h"
#include "ios/chrome/browser/passwords/password_check_observer_bridge.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details_consumer.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details_table_view_controller_delegate.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

using CompromisedCredentialsView =
    password_manager::CompromisedCredentialsManager::CredentialsView;

@interface PasswordDetailsMediator () <
    PasswordCheckObserver,
    PasswordDetailsTableViewControllerDelegate> {
  // Password Check manager.
  IOSChromePasswordCheckManager* _manager;

  // Listens to compromised passwords changes.
  std::unique_ptr<PasswordCheckObserverBridge> _passwordCheckObserver;
}

@end

@implementation PasswordDetailsMediator

- (instancetype)initWithPassword:(const autofill::PasswordForm&)passwordForm
            passwordCheckManager:(IOSChromePasswordCheckManager*)manager {
  self = [super init];
  if (self) {
    _manager = manager;
    _password = passwordForm;
    _passwordCheckObserver.reset(
        new PasswordCheckObserverBridge(self, manager));
  }
  return self;
}

- (void)setConsumer:(id<PasswordDetailsConsumer>)consumer {
  if (_consumer == consumer)
    return;
  _consumer = consumer;

  [self fetchPasswordWith:_manager->GetCompromisedCredentials()];
}

- (void)disconnect {
  _manager->RemoveObserver(_passwordCheckObserver.get());
}

#pragma mark - PasswordDetailsTableViewControllerDelegate

- (void)passwordDetailsViewController:
            (PasswordDetailsTableViewController*)viewController
               didEditPasswordDetails:(PasswordDetails*)password {
  if ([password.password length] != 0) {
    password.compromised
        ? _manager->EditCompromisedPasswordForm(
              _password, base::SysNSStringToUTF8(password.password))
        : _manager->EditPasswordForm(
              _password, base::SysNSStringToUTF8(password.password));
    _password.password_value = base::SysNSStringToUTF16(password.password);
  } else {
    [self fetchPasswordWith:_manager->GetCompromisedCredentials()];
  }
}

#pragma mark - PasswordCheckObserver

- (void)passwordCheckStateDidChange:(PasswordCheckState)state {
  // No-op. Changing password check state has no effect on compromised
  // passwords.
}

- (void)compromisedCredentialsDidChange:
    (CompromisedCredentialsView)credentials {
  [self fetchPasswordWith:credentials];
}

#pragma mark - Private

// Updates password details and sets it to a consumer.
- (void)fetchPasswordWith:(CompromisedCredentialsView)credentials {
  PasswordDetails* password =
      [[PasswordDetails alloc] initWithPasswordForm:_password];
  password.compromised = NO;

  for (const auto& credential : credentials) {
    if (std::tie(credential.signon_realm, credential.username,
                 credential.password) == std::tie(_password.signon_realm,
                                                  _password.username_value,
                                                  _password.password_value))
      password.compromised = YES;
  }

  [self.consumer setPassword:password];
}

@end
