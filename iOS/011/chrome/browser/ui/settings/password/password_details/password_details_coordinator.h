// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IOS_CHROME_BROWSER_UI_SETTINGS_PASSWORD_PASSWORD_DETAILS_PASSWORD_DETAILS_COORDINATOR_H_
#define IOS_CHROME_BROWSER_UI_SETTINGS_PASSWORD_PASSWORD_DETAILS_PASSWORD_DETAILS_COORDINATOR_H_

#import "ios/chrome/browser/ui/coordinators/chrome_coordinator.h"

@protocol ApplicationCommands;
class Browser;
class IOSChromePasswordCheckManager;
@protocol PasswordDetailsCoordinatorDelegate;
@class ReauthenticationModule;

namespace autofill {
struct PasswordForm;
}

// This coordinator presents a password details for the user.
@interface PasswordDetailsCoordinator : ChromeCoordinator

- (instancetype)
    initWithBaseNavigationController:
        (UINavigationController*)navigationController
                             browser:(Browser*)browser
                            password:(const autofill::PasswordForm&)password
                        reauthModule:(ReauthenticationModule*)reauthModule
                passwordCheckManager:(IOSChromePasswordCheckManager*)manager
    NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithBaseViewController:(UIViewController*)viewController
                                   browser:(Browser*)browser NS_UNAVAILABLE;

// Delegate.
@property(nonatomic, weak) id<PasswordDetailsCoordinatorDelegate> delegate;

// Dispatcher.
@property(nonatomic, weak) id<ApplicationCommands> dispatcher;

@end

#endif  // IOS_CHROME_BROWSER_UI_SETTINGS_PASSWORD_PASSWORD_DETAILS_PASSWORD_DETAILS_COORDINATOR_H_
