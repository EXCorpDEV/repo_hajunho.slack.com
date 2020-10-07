// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/settings/safety_check/safety_check_mediator.h"

#include "base/mac/foundation_util.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/sys_string_conversions.h"
#include "base/strings/utf_string_conversions.h"
#include "components/password_manager/core/browser/leak_detection_dialog_utils.h"
#include "components/password_manager/core/common/password_manager_features.h"
#include "components/prefs/pref_service.h"
#include "components/safe_browsing/core/common/safe_browsing_prefs.h"
#include "components/safe_browsing/core/features.h"
#include "ios/chrome/browser/passwords/ios_chrome_password_check_manager.h"
#include "ios/chrome/browser/passwords/ios_chrome_password_check_manager_factory.h"
#include "ios/chrome/browser/passwords/password_check_observer_bridge.h"
#include "ios/chrome/browser/passwords/password_store_observer_bridge.h"
#include "ios/chrome/browser/pref_names.h"
#import "ios/chrome/browser/signin/authentication_service.h"
#include "ios/chrome/browser/sync/sync_setup_service.h"
#import "ios/chrome/browser/ui/settings/cells/settings_check_item.h"
#import "ios/chrome/browser/ui/settings/safety_check/safety_check_consumer.h"
#import "ios/chrome/browser/ui/settings/safety_check/safety_check_navigation_commands.h"
#import "ios/chrome/browser/ui/settings/safety_check/safety_check_table_view_controller.h"
#import "ios/chrome/browser/ui/settings/utils/observable_boolean.h"
#import "ios/chrome/browser/ui/settings/utils/pref_backed_boolean.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_cells_constants.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_link_header_footer_item.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_text_item.h"
#import "ios/chrome/browser/ui/ui_feature_flags.h"
#import "ios/chrome/browser/ui/util/uikit_ui_util.h"
#import "ios/chrome/common/string_util.h"
#import "ios/chrome/common/ui/colors/UIColor+cr_semantic_colors.h"
#import "ios/chrome/common/ui/colors/semantic_color_names.h"
#include "ios/chrome/grit/ios_chromium_strings.h"
#include "ios/chrome/grit/ios_strings.h"
#import "net/base/mac/url_conversions.h"
#include "ui/base/l10n/l10n_util.h"
#include "url/gurl.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

using l10n_util::GetNSString;
using safe_browsing::kSafeBrowsingAvailableOnIOS;

constexpr char kSafeBrowsingStringURL[] = "chrome://settings/syncSetup";

namespace {

typedef NSArray<TableViewItem*>* ItemArray;

typedef NS_ENUM(NSInteger, ItemType) {
  // CheckTypes section.
  UpdateItemType = kItemTypeEnumZero,
  PasswordItemType,
  SafeBrowsingItemType,
  HeaderItem,
  // CheckStart section.
  CheckStartItemType,
};

// Enum with all possible states of the update check.
typedef NS_ENUM(NSInteger, UpdateCheckRowStates) {
  // When the user is up to date.
  UpdateCheckRowStateUpToDate,
  // When the check has not been run yet.
  UpdateCheckRowStateDefault,
  // When the user is out of date.
  UpdateCheckRowStateOutOfDate,
  // When the user is managed.
  UpdateCheckRowStateManaged,
  // When the check is running.
  UpdateCheckRowStateRunning,
};

// Enum with all possible states of the password check.
typedef NS_ENUM(NSInteger, PasswordCheckRowStates) {
  // When no compromised passwords were detected.
  PasswordCheckRowStateSafe,
  // When user has compromised passwords.
  PasswordCheckRowStateUnSafe,
  // When check has not been run yet.
  PasswordCheckRowStateDefault,
  // When password check is running.
  PasswordCheckRowStateRunning,
  // When user has no passwords and check can't be performed.
  PasswordCheckRowStateDisabled,
  // When password check failed due to network issues, quota limit or others.
  PasswordCheckRowStateError,
};

// Enum with all possible states of the Safe Browsing check.
typedef NS_ENUM(NSInteger, SafeBrowsingCheckRowStates) {
  // When check was not run yet.
  SafeBrowsingCheckRowStateDefault,
  // When Safe Browsing is managed by admin.
  SafeBrowsingCheckRowStateManaged,
  // When the Safe Browsing check is running.
  SafeBrowsingCheckRowStateRunning,
  // When Safe Browsing is enabled.
  SafeBrowsingCheckRowStateSafe,
  // When Safe Browsing is disabled.
  SafeBrowsingCheckRowStateUnsafe,
};

// Enum with all possible states of the button to start the check.
typedef NS_ENUM(NSInteger, CheckStartStates) {
  // When the check is not running.
  CheckStartStateDefault,
  // When the check is running.
  CheckStartStateCancel,
};

}  // namespace

@interface SafetyCheckMediator () <BooleanObserver, PasswordCheckObserver> {
  // A helper object for observing changes in the password check status
  // and changes to the compromised credentials list.
  std::unique_ptr<PasswordCheckObserverBridge> _passwordCheckObserver;
}

// Header for the Safety Check page.
@property(nonatomic, strong) TableViewLinkHeaderFooterItem* headerItem;

// SettingsCheckItem used to display the state of the Safe Browsing check.
@property(nonatomic, strong) SettingsCheckItem* safeBrowsingCheckItem;

// Current state of the Safe Browsing check.
@property(nonatomic, assign)
    SafeBrowsingCheckRowStates safeBrowsingCheckRowState;

// SettingsCheckItem used to display the state of the update check.
@property(nonatomic, strong) SettingsCheckItem* updateCheckItem;

// Current state of the update check.
@property(nonatomic, assign) UpdateCheckRowStates updateCheckRowState;

// SettingsCheckItem used to display the state of the password check.
@property(nonatomic, strong) SettingsCheckItem* passwordCheckItem;

// Current state of the password check.
@property(nonatomic, assign) PasswordCheckRowStates passwordCheckRowState;

// Row button to start the safety check.
@property(nonatomic, strong) TableViewTextItem* checkStartItem;

// Current state of the start safety check row button.
@property(nonatomic, assign) CheckStartStates checkStartState;

// Preference value for Safe Browsing.
@property(nonatomic, strong, readonly)
    PrefBackedBoolean* safeBrowsingPreference;

// If the Safe Browsing preference is managed.
@property(nonatomic, assign) BOOL safeBrowsingPreferenceManaged;

// The service responsible for password check feature.
@property(nonatomic, assign) scoped_refptr<IOSChromePasswordCheckManager>
    passwordCheckManager;

// Current state of password check.
@property(nonatomic, assign) PasswordCheckState currentPasswordCheckState;

// If any checks in safety check are still running.
@property(nonatomic, assign, readonly) BOOL checksRemaining;

// Service used to check if user is signed in.
@property(nonatomic, assign) AuthenticationService* authService;

// Service to check if passwords are synced.
@property(nonatomic, assign) SyncSetupService* syncService;

@end

@implementation SafetyCheckMediator

- (instancetype)initWithUserPrefService:(PrefService*)userPrefService
                   passwordCheckManager:
                       (scoped_refptr<IOSChromePasswordCheckManager>)
                           passwordCheckManager
                            authService:(AuthenticationService*)authService
                            syncService:(SyncSetupService*)syncService {
  self = [super init];
  if (self) {
    DCHECK(userPrefService);
    DCHECK(passwordCheckManager);
    DCHECK(authService);
    DCHECK(syncService);

    _authService = authService;
    _syncService = syncService;

    _passwordCheckManager = passwordCheckManager;
    _currentPasswordCheckState = _passwordCheckManager->GetPasswordCheckState();

    _passwordCheckObserver = std::make_unique<PasswordCheckObserverBridge>(
        self, _passwordCheckManager.get());

    _safeBrowsingPreference = [[PrefBackedBoolean alloc]
        initWithPrefService:userPrefService
                   prefName:prefs::kSafeBrowsingEnabled];
    _safeBrowsingPreference.observer = self;
    _safeBrowsingPreferenceManaged =
        userPrefService->IsManagedPreference(prefs::kSafeBrowsingEnabled);

    _headerItem =
        [[TableViewLinkHeaderFooterItem alloc] initWithType:HeaderItem];
    _headerItem.text =
        l10n_util::GetNSString(IDS_IOS_SETTINGS_SAFETY_CHECK_PAGE_HEADER);

    _updateCheckRowState = UpdateCheckRowStateDefault;
    _updateCheckItem = [[SettingsCheckItem alloc] initWithType:UpdateItemType];
    _updateCheckItem.text =
        l10n_util::GetNSString(IDS_IOS_SETTINGS_SAFETY_CHECK_UPDATES_TITLE);
    UIImage* updateCheckIcon = [[UIImage imageNamed:@"settings_info"]
        imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    _updateCheckItem.leadingImage = updateCheckIcon;
    _updateCheckItem.leadingImageTintColor = [UIColor colorNamed:kGrey400Color];
    _updateCheckItem.enabled = YES;
    _updateCheckItem.indicatorHidden = YES;
    _updateCheckItem.infoButtonHidden = YES;
    _updateCheckItem.trailingImage = nil;

    _passwordCheckRowState = PasswordCheckRowStateDefault;
    _passwordCheckItem =
        [[SettingsCheckItem alloc] initWithType:PasswordItemType];
    _passwordCheckItem.text =
        l10n_util::GetNSString(IDS_IOS_SETTINGS_SAFETY_CHECK_PASSWORDS_TITLE);
    UIImage* passwordCheckIcon = [[UIImage imageNamed:@"password_key"]
        imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    _passwordCheckItem.leadingImage = passwordCheckIcon;
    _passwordCheckItem.leadingImageTintColor =
        [UIColor colorNamed:kGrey400Color];
    _passwordCheckItem.enabled = YES;
    _passwordCheckItem.indicatorHidden = YES;
    _passwordCheckItem.infoButtonHidden = YES;
    _passwordCheckItem.trailingImage = nil;

    _safeBrowsingCheckRowState = SafeBrowsingCheckRowStateDefault;
    _safeBrowsingCheckItem =
        [[SettingsCheckItem alloc] initWithType:SafeBrowsingItemType];
    _safeBrowsingCheckItem.text = l10n_util::GetNSString(
        IDS_IOS_SETTINGS_SAFETY_CHECK_SAFE_BROWSING_TITLE);
    UIImage* safeBrowsingCheckIcon =
        [[UIImage imageNamed:@"settings_safe_browsing"]
            imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    _safeBrowsingCheckItem.leadingImage = safeBrowsingCheckIcon;
    _safeBrowsingCheckItem.leadingImageTintColor =
        [UIColor colorNamed:kGrey400Color];
    _safeBrowsingCheckItem.enabled = YES;
    _safeBrowsingCheckItem.indicatorHidden = YES;
    _safeBrowsingCheckItem.infoButtonHidden = YES;
    _safeBrowsingCheckItem.trailingImage = nil;

    _checkStartState = CheckStartStateDefault;
    _checkStartItem =
        [[TableViewTextItem alloc] initWithType:CheckStartItemType];
    _checkStartItem.text = GetNSString(IDS_IOS_CHECK_PASSWORDS_NOW_BUTTON);
    _checkStartItem.textColor = [UIColor colorNamed:kBlueColor];
  }
  return self;
}

- (void)setConsumer:(id<SafetyCheckConsumer>)consumer {
  if (_consumer == consumer)
    return;
  _consumer = consumer;
  NSArray* checkItems = @[
    self.updateCheckItem, self.passwordCheckItem, self.safeBrowsingCheckItem
  ];
  [_consumer setCheckItems:checkItems];
  [_consumer setSafetyCheckHeaderItem:self.headerItem];
  [_consumer setCheckStartItem:self.checkStartItem];
}

#pragma mark - PasswordCheckObserver

- (void)passwordCheckStateDidChange:(PasswordCheckState)state {
  if (state == self.currentPasswordCheckState)
    return;

  self.passwordCheckRowState = [self computePasswordCheckRowState:state];
  // Push update to the display.
  [self reconfigurePasswordCheckItem];
}

- (void)compromisedCredentialsDidChange:
    (password_manager::CompromisedCredentialsManager::CredentialsView)
        credentials {
  self.passwordCheckRowState =
      [self computePasswordCheckRowState:self.currentPasswordCheckState];
  // Push update to the display.
  [self reconfigurePasswordCheckItem];
}

#pragma mark - SafetyCheckServiceDelegate

- (void)didSelectItem:(TableViewItem*)item {
  ItemType type = static_cast<ItemType>(item.type);
  switch (type) {
    case UpdateItemType: {
      switch (self.updateCheckRowState) {
        case UpdateCheckRowStateDefault:   // No tap action.
        case UpdateCheckRowStateRunning:   // No tap action.
        case UpdateCheckRowStateUpToDate:  // No tap action.
          break;
        case UpdateCheckRowStateManaged:  // Managed state popover handled by
                                          // coordinator.
          break;
        case UpdateCheckRowStateOutOfDate:  // Link to app store.
          break;
      }
      break;
    }
    case PasswordItemType: {
      switch (self.passwordCheckRowState) {
        case PasswordCheckRowStateDefault:  // No tap action.
        case PasswordCheckRowStateRunning:  // No tap action.
        case PasswordCheckRowStateSafe:     // No tap action.
          break;
        case PasswordCheckRowStateUnSafe:
          [self.handler showPasswordIssuesPage];
          break;
        case PasswordCheckRowStateDisabled:  // Popover handled by coordinator.
        case PasswordCheckRowStateError:     // Popover handled by coordinator.
          break;
      }
      break;
    }
    case SafeBrowsingItemType: {
      switch (self.safeBrowsingCheckRowState) {
        case SafeBrowsingCheckRowStateDefault:  // No tap action.
        case SafeBrowsingCheckRowStateRunning:  // No tap action.
        case SafeBrowsingCheckRowStateSafe:     // No tap action.
          break;
        case SafeBrowsingCheckRowStateManaged:  // Managed state popover handled
                                                // by coordinator.
          break;
        case SafeBrowsingCheckRowStateUnsafe:
          [self.handler showSafeBrowsingPreferencePage];
          break;
      }
      break;
    }
    case CheckStartItemType: {
      [self checkStartOrCancel];
      break;
    }
    case HeaderItem:
      break;
  }
}

- (BOOL)isItemWithErrorInfo:(TableViewItem*)item {
  ItemType type = static_cast<ItemType>(item.type);
  return (type != CheckStartItemType);
}

- (void)infoButtonWasTapped:(UIButton*)buttonView
              usingItemType:(NSInteger)itemType {
  // Show the managed popover if needed.
  if (self.safeBrowsingCheckRowState == SafeBrowsingCheckRowStateManaged) {
    [self.handler showManagedInfoFrom:buttonView];
    return;
  }
  NSAttributedString* info = [self getPopoverInfoForType:itemType];

  // If |info| is empty there is no popover to display.
  if (!info)
    return;

  // Push popover to coordinator.
  [self.handler showErrorInfoFrom:buttonView withText:info];
}

#pragma mark - BooleanObserver

- (void)booleanDidChange:(id<ObservableBoolean>)observableBoolean {
  // TODO(crbug.com/1078782): Handle safe browsing state changes to reward user
  // for fixing state.
  return;
}

#pragma mark - Private methods

// Computes the text needed for a popover on |itemType| if available.
- (NSAttributedString*)getPopoverInfoForType:(NSInteger)itemType {
  ItemType type = static_cast<ItemType>(itemType);
  switch (type) {
    case PasswordItemType:
      return [self passwordCheckErrorInfo];
    case SafeBrowsingItemType: {
      NSString* message = l10n_util::GetNSString(
          IDS_IOS_SETTINGS_SAFETY_CHECK_SAFE_BROWSING_DISABLED_INFO);
      GURL safeBrowsingURL(kSafeBrowsingStringURL);
      return [self attributedStringWithText:message link:safeBrowsingURL];
    }
    case UpdateItemType:
    case CheckStartItemType:
    case HeaderItem:
      return nil;
  }
}

// Computes the appropriate display state of the password check row based on
// currentPasswordCheckState.
- (PasswordCheckRowStates)computePasswordCheckRowState:
    (PasswordCheckState)newState {
  BOOL wasRunning =
      self.currentPasswordCheckState == PasswordCheckState::kRunning;
  self.currentPasswordCheckState = newState;

  switch (self.currentPasswordCheckState) {
    case PasswordCheckState::kRunning:
      return PasswordCheckRowStateRunning;
    case PasswordCheckState::kNoPasswords:
      return PasswordCheckRowStateDefault;
    case PasswordCheckState::kSignedOut:
    case PasswordCheckState::kOffline:
    case PasswordCheckState::kQuotaLimit:
    case PasswordCheckState::kOther:
      return self.passwordCheckManager->GetCompromisedCredentials().empty()
                 ? PasswordCheckRowStateError
                 : PasswordCheckRowStateUnSafe;
    case PasswordCheckState::kCanceled:
    case PasswordCheckState::kIdle: {
      if (!self.passwordCheckManager->GetCompromisedCredentials().empty()) {
        return PasswordCheckRowStateUnSafe;
      } else if (self.currentPasswordCheckState == PasswordCheckState::kIdle) {
        // Safe state is only possible after the state transitioned from
        // kRunning to kIdle.
        return (wasRunning) ? PasswordCheckRowStateSafe
                            : PasswordCheckRowStateDefault;
      }
      return PasswordCheckRowStateDefault;
    }
  }
}

// Computes the appropriate error info to be displayed in the passwords popover.
- (NSAttributedString*)passwordCheckErrorInfo {
  if (!self.passwordCheckManager->GetCompromisedCredentials().empty())
    return nil;

  NSString* message;
  GURL linkURL;

  switch (self.currentPasswordCheckState) {
    case PasswordCheckState::kRunning:
    case PasswordCheckState::kNoPasswords:
      message =
          l10n_util::GetNSString(IDS_IOS_PASSWORD_CHECK_ERROR_NO_PASSWORDS);
      break;
    case PasswordCheckState::kCanceled:
    case PasswordCheckState::kIdle:
      return nil;
    case PasswordCheckState::kSignedOut:
      message = l10n_util::GetNSString(IDS_IOS_PASSWORD_CHECK_ERROR_SIGNED_OUT);
      break;
    case PasswordCheckState::kOffline:
      message = l10n_util::GetNSString(IDS_IOS_PASSWORD_CHECK_ERROR_OFFLINE);
      break;
    case PasswordCheckState::kQuotaLimit:
      if ([self canUseAccountPasswordCheckup]) {
        message = l10n_util::GetNSString(
            IDS_IOS_PASSWORD_CHECK_ERROR_QUOTA_LIMIT_VISIT_GOOGLE);
        linkURL = password_manager::GetPasswordCheckupURL(
            password_manager::PasswordCheckupReferrer::kPasswordCheck);
      } else {
        message =
            l10n_util::GetNSString(IDS_IOS_PASSWORD_CHECK_ERROR_QUOTA_LIMIT);
      }
      break;
    case PasswordCheckState::kOther:
      message = l10n_util::GetNSString(IDS_IOS_PASSWORD_CHECK_ERROR_OTHER);
      break;
  }
  return [self attributedStringWithText:message link:linkURL];
}

// Computes whether user is capable to run password check in Google Account.
- (BOOL)canUseAccountPasswordCheckup {
  return (self.authService->IsAuthenticated() &&
          self.authService->GetAuthenticatedIdentity()) &&
         (self.syncService->IsSyncEnabled() &&
          !self.syncService->IsEncryptEverythingEnabled());
}

// Configures check error info with a link for popovers.
- (NSAttributedString*)attributedStringWithText:(NSString*)text
                                           link:(GURL)link {
  NSRange range;

  NSString* strippedText = ParseStringWithLink(text, &range);

  NSRange fullRange = NSMakeRange(0, strippedText.length);
  NSMutableAttributedString* attributedText =
      [[NSMutableAttributedString alloc] initWithString:strippedText];
  [attributedText addAttribute:NSForegroundColorAttributeName
                         value:[UIColor colorNamed:kTextSecondaryColor]
                         range:fullRange];

  [attributedText
      addAttribute:NSFontAttributeName
             value:[UIFont preferredFontForTextStyle:UIFontTextStyleSubheadline]
             range:fullRange];

  if (range.location != NSNotFound && range.length != 0) {
    NSURL* URL = net::NSURLWithGURL(link);
    id linkValue = URL ? URL : @"";
    [attributedText addAttribute:NSLinkAttributeName
                           value:linkValue
                           range:range];
  }
  return attributedText;
}

// Upon a tap of checkStartItem either starts or cancels a safety check.
- (void)checkStartOrCancel {
  // If a check is already running cancel it.
  if (self.checksRemaining) {
    // Reset check items to default.
    self.updateCheckRowState = UpdateCheckRowStateDefault;
    self.passwordCheckRowState = PasswordCheckRowStateDefault;
    self.safeBrowsingCheckRowState = SafeBrowsingCheckRowStateDefault;

    // Change checkStartItem to default state.
    self.checkStartState = CheckStartStateDefault;

    // Stop any running checks.
    self.passwordCheckManager->StopPasswordCheck();

  } else {
    // Otherwise start a check.

    // Set check items to spinning wheel.
    self.updateCheckRowState = UpdateCheckRowStateRunning;
    self.passwordCheckRowState = PasswordCheckRowStateRunning;
    self.safeBrowsingCheckRowState = SafeBrowsingCheckRowStateRunning;

    // Change checkStartItem to cancel state.
    self.checkStartState = CheckStartStateCancel;
  }

  // Update the display.
  [self reconfigureUpdateCheckItem];
  [self reconfigurePasswordCheckItem];
  [self reconfigureSafeBrowsingCheckItem];
  [self reconfigureCheckStartSection];

  // The display should be changed to loading icons before any checks are
  // started.
  if (self.checksRemaining) {
    // This handles a discrepancy between password check and safety check.  In
    // password check a user cannot start a check if they have no passwords, but
    // in safety check they can, but the |passwordCheckManager| won't even start
    // a check. This if block below allows safety check to push the disabled
    // state after check now is pressed.
    if (self.currentPasswordCheckState == PasswordCheckState::kNoPasswords) {
      self.passwordCheckRowState = PasswordCheckRowStateDisabled;
      // Want to show the loading wheel momentarily.
      dispatch_after(
          dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.5 * NSEC_PER_SEC)),
          dispatch_get_main_queue(), ^{
            // Check if the check was cancelled while waiting.
            if (self.checksRemaining)
              [self reconfigurePasswordCheckItem];
          });
    } else {
      self.passwordCheckManager->StartPasswordCheck();
    }
    // Want to show the loading wheel momentarily.
    dispatch_after(
        dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.75 * NSEC_PER_SEC)),
        dispatch_get_main_queue(), ^{
          // Check if the check was cancelled while waiting.
          if (self.checksRemaining)
            [self checkAndReconfigureSafeBrowsingState];
        });
  }
}

// Checks if any of the safety checks are still running, resets |checkStartItem|
// if all checks have finished..
- (void)resetsCheckStartItemIfNeeded {
  if (self.checksRemaining) {
    return;
  }
  // If no checks are still running, reset |checkStartItem|.
  self.checkStartState = CheckStartStateDefault;
  [self reconfigureCheckStartSection];
  return;
}

// Computes if any of the safety checks are still running.
- (BOOL)checksRemaining {
  BOOL passwordCheckRunning =
      self.passwordCheckRowState == PasswordCheckRowStateRunning;
  BOOL safeBrowsingCheckRunning =
      self.safeBrowsingCheckRowState == SafeBrowsingCheckRowStateRunning;
  BOOL updateCheckRunning =
      self.updateCheckRowState == UpdateCheckRowStateRunning;
  return updateCheckRunning || passwordCheckRunning || safeBrowsingCheckRunning;
}

// Performs the Safe Browsing check and triggers the display update/
- (void)checkAndReconfigureSafeBrowsingState {
  if (!self.safeBrowsingPreferenceManaged) {
    self.safeBrowsingCheckRowState = self.safeBrowsingPreference.value
                                         ? SafeBrowsingCheckRowStateSafe
                                         : SafeBrowsingCheckRowStateUnsafe;
  }
  if (self.safeBrowsingCheckRowState == SafeBrowsingCheckRowStateUnsafe &&
      self.safeBrowsingPreferenceManaged) {
    self.safeBrowsingCheckRowState = SafeBrowsingCheckRowStateManaged;
  }
  // See if this was the last test.
  [self resetsCheckStartItemIfNeeded];

  // Update |safeBrowsingCheckItem| display.
  [self reconfigureSafeBrowsingCheckItem];
}

// Reconfigures the display of the |updateCheckItem| based on current state of
// |updateCheckRowState|.
- (void)reconfigureUpdateCheckItem {
  // Reset state to prevent conflicts.
  self.updateCheckItem.enabled = YES;
  self.updateCheckItem.indicatorHidden = YES;
  self.updateCheckItem.infoButtonHidden = YES;
  self.updateCheckItem.detailText = nil;
  self.updateCheckItem.trailingImage = nil;
  self.updateCheckItem.trailingImageTintColor = nil;

  // On any item update, see if any checks are still running.
  [self resetsCheckStartItemIfNeeded];

  switch (self.updateCheckRowState) {
    case UpdateCheckRowStateDefault:
      break;
    case UpdateCheckRowStateRunning: {
      self.updateCheckItem.indicatorHidden = NO;
      break;
    }
    case UpdateCheckRowStateManaged:
    case UpdateCheckRowStateUpToDate:
    case UpdateCheckRowStateOutOfDate:
      break;
  }

  [self.consumer reconfigureCellsForItems:@[ self.updateCheckItem ]];
}

// Reconfigures the display of the |passwordCheckItem| based on current state of
// |passwordCheckRowState|.
- (void)reconfigurePasswordCheckItem {
  // Reset state to prevent conflicts.
  self.passwordCheckItem.enabled = YES;
  self.passwordCheckItem.indicatorHidden = YES;
  self.passwordCheckItem.infoButtonHidden = YES;
  self.passwordCheckItem.detailText = nil;
  self.passwordCheckItem.trailingImage = nil;
  self.passwordCheckItem.trailingImageTintColor = nil;
  self.passwordCheckItem.accessoryType = UITableViewCellAccessoryNone;

  // On any item update, see if any checks are still running.
  [self resetsCheckStartItemIfNeeded];

  switch (self.passwordCheckRowState) {
    case PasswordCheckRowStateDefault:
      break;
    case PasswordCheckRowStateRunning: {
      self.passwordCheckItem.indicatorHidden = NO;
      break;
    }
    case PasswordCheckRowStateSafe: {
      DCHECK(self.passwordCheckManager->GetCompromisedCredentials().empty());
      UIImage* safeIconImage = [[UIImage imageNamed:@"settings_safe_state"]
          imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
      self.passwordCheckItem.detailText =
          base::SysUTF16ToNSString(l10n_util::GetPluralStringFUTF16(
              IDS_IOS_CHECK_PASSWORDS_COMPROMISED_COUNT, 0));
      self.passwordCheckItem.trailingImage = safeIconImage;
      self.passwordCheckItem.trailingImageTintColor =
          [UIColor colorNamed:kGreenColor];
      break;
    }
    case PasswordCheckRowStateUnSafe: {
      self.passwordCheckItem.detailText =
          base::SysUTF16ToNSString(l10n_util::GetPluralStringFUTF16(
              IDS_IOS_CHECK_PASSWORDS_COMPROMISED_COUNT,
              self.passwordCheckManager->GetCompromisedCredentials().size()));
      UIImage* unSafeIconImage = [[UIImage imageNamed:@"settings_unsafe_state"]
          imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
      self.passwordCheckItem.trailingImage = unSafeIconImage;
      self.passwordCheckItem.trailingImageTintColor =
          [UIColor colorNamed:kRedColor];
      self.passwordCheckItem.accessoryType =
          UITableViewCellAccessoryDisclosureIndicator;
      break;
    }
    case PasswordCheckRowStateDisabled:
    case PasswordCheckRowStateError: {
      self.passwordCheckItem.detailText =
          l10n_util::GetNSString(IDS_IOS_PASSWORD_CHECK_ERROR);
      self.passwordCheckItem.infoButtonHidden = NO;
      break;
    }
  }

  [self.consumer reconfigureCellsForItems:@[ self.passwordCheckItem ]];
}

// Reconfigures the display of the |safeBrowsingCheckItem| based on current
// state of |safeBrowsingCheckRowState|.
- (void)reconfigureSafeBrowsingCheckItem {
  // Reset state to prevent conflicts.
  self.safeBrowsingCheckItem.enabled = YES;
  self.safeBrowsingCheckItem.indicatorHidden = YES;
  self.safeBrowsingCheckItem.infoButtonHidden = YES;
  self.safeBrowsingCheckItem.detailText = nil;
  self.safeBrowsingCheckItem.trailingImage = nil;
  self.safeBrowsingCheckItem.trailingImageTintColor = nil;

  // On any item update, see if any checks are still running.
  [self resetsCheckStartItemIfNeeded];

  switch (self.safeBrowsingCheckRowState) {
    case SafeBrowsingCheckRowStateDefault:
      break;
    case SafeBrowsingCheckRowStateRunning: {
      self.safeBrowsingCheckItem.indicatorHidden = NO;
      break;
    }
    case SafeBrowsingCheckRowStateManaged: {
      self.safeBrowsingCheckItem.infoButtonHidden = NO;
      self.safeBrowsingCheckItem.detailText =
          GetNSString(IDS_IOS_SETTINGS_SAFETY_CHECK_SAFE_BROWSING_MANAGED_DESC);
      break;
    }
    case SafeBrowsingCheckRowStateSafe: {
      UIImage* safeIconImage = [[UIImage imageNamed:@"settings_safe_state"]
          imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
      self.safeBrowsingCheckItem.trailingImage = safeIconImage;
      self.safeBrowsingCheckItem.trailingImageTintColor =
          [UIColor colorNamed:kGreenColor];
      self.safeBrowsingCheckItem.detailText =
          GetNSString(IDS_IOS_SETTINGS_SAFETY_CHECK_SAFE_BROWSING_ENABLED_DESC);
      break;
    }
    case SafeBrowsingCheckRowStateUnsafe: {
      self.safeBrowsingCheckItem.infoButtonHidden = NO;
      self.safeBrowsingCheckItem.detailText = GetNSString(
          IDS_IOS_SETTINGS_SAFETY_CHECK_SAFE_BROWSING_DISABLED_DESC);
      break;
    }
  }

  [self.consumer reconfigureCellsForItems:@[ self.safeBrowsingCheckItem ]];
}

// Updates the display of checkStartItem based on its current state.
- (void)reconfigureCheckStartSection {
  if (self.checkStartState == CheckStartStateDefault) {
    self.checkStartItem.text = GetNSString(IDS_IOS_CHECK_PASSWORDS_NOW_BUTTON);
  } else {
    self.checkStartItem.text =
        GetNSString(IDS_IOS_CANCEL_PASSWORD_CHECK_BUTTON);
  }
  [self.consumer reconfigureCellsForItems:@[ self.checkStartItem ]];
}

@end
