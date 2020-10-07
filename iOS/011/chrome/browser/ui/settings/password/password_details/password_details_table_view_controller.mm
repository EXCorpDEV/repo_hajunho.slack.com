// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/settings/password/password_details/password_details_table_view_controller.h"

#include "base/ios/ios_util.h"
#include "base/mac/foundation_util.h"
#include "base/metrics/histogram_macros.h"
#include "base/strings/sys_string_conversions.h"
#include "components/password_manager/core/browser/password_manager_metrics_util.h"
#import "ios/chrome/browser/ui/commands/application_commands.h"
#import "ios/chrome/browser/ui/commands/open_new_tab_command.h"
#import "ios/chrome/browser/ui/settings/cells/settings_image_detail_text_item.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details_consumer.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details_handler.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details_table_view_constants.h"
#import "ios/chrome/browser/ui/settings/password/password_details/password_details_table_view_controller_delegate.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_cells_constants.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_text_edit_item.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_text_item.h"
#include "ios/chrome/browser/ui/util/uikit_ui_util.h"
#import "ios/chrome/common/ui/colors/UIColor+cr_semantic_colors.h"
#import "ios/chrome/common/ui/colors/semantic_color_names.h"
#import "ios/chrome/common/ui/reauthentication/reauthentication_module.h"
#include "ios/chrome/grit/ios_chromium_strings.h"
#include "ios/chrome/grit/ios_strings.h"
#include "ui/base/l10n/l10n_util_mac.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace {

using password_manager::metrics_util::LogPasswordSettingsReauthResult;
using password_manager::metrics_util::ReauthResult;

// Padding used between the image and the text labels.
const CGFloat kWarningIconSize = 20;

typedef NS_ENUM(NSInteger, SectionIdentifier) {
  SectionIdentifierPassword = kSectionIdentifierEnumZero,
  SectionIdentifierCompromisedInfo
};

typedef NS_ENUM(NSInteger, ItemType) {
  ItemTypeWebsite = kItemTypeEnumZero,
  ItemTypeUsername,
  ItemTypePassword,
  ItemTypeChangePasswordButton,
  ItemTypeChangePasswordRecommendation,
};

typedef NS_ENUM(NSInteger, ReauthenticationReason) {
  ReauthenticationReasonShow = 0,
  ReauthenticationReasonCopy,
  ReauthenticationReasonEdit,
};

}  // namespace

@interface PasswordDetailsTableViewController ()

// Password which is shown on the screen.
@property(nonatomic, strong) PasswordDetails* password;

// Whether the password is shown in plain text form or in masked form.
@property(nonatomic, assign, getter=isPasswordShown) BOOL passwordShown;

// The text item related to the password value.
@property(nonatomic, strong) TableViewTextEditItem* passwordTextItem;

@end

@implementation PasswordDetailsTableViewController

#pragma mark - UIViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.tableView.accessibilityIdentifier = kPasswordDetailsViewControllerId;
  self.tableView.allowsSelectionDuringEditing = YES;

  UILabel* titleLabel = [[UILabel alloc] init];
  titleLabel.lineBreakMode = NSLineBreakByTruncatingHead;
  titleLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
  titleLabel.adjustsFontForContentSizeCategory = YES;
  titleLabel.text = self.password.origin;
  self.navigationItem.titleView = titleLabel;
}

- (void)viewDidDisappear:(BOOL)animated {
  [self.handler passwordDetailsTableViewControllerDidDisappear];
  [super viewDidDisappear:animated];
}

#pragma mark - ChromeTableViewController

- (void)editButtonPressed {
  // If password value is missing, proceed with editing without
  // reauthentication.
  if (![self.password.password length]) {
    [super editButtonPressed];
    return;
  }

  // Request reauthentication before revealing password during editing.
  // Editing mode will be entered on successful reauth.
  if (!self.tableView.editing && !self.isPasswordShown) {
    [self attemptToShowPasswordFor:ReauthenticationReasonEdit];
    return;
  }

  if (self.tableView.editing) {
    // If password value was changed show confirmation dialog before saving
    // password. Editing mode will be exited only if user confirm saving.
    if (self.password.password != self.passwordTextItem.textFieldValue) {
      [self.handler showPasswordEditDialogWithOrigin:self.password.origin];
      return;
    }
  }

  [super editButtonPressed];
  [self reloadData];
}

- (void)loadModel {
  [super loadModel];

  TableViewModel* model = self.tableViewModel;
  [model addSectionWithIdentifier:SectionIdentifierPassword];

  [model addItem:[self websiteItem]
      toSectionWithIdentifier:SectionIdentifierPassword];

  // Blocked password forms don't have username value.
  if ([self.password.username length]) {
    [model addItem:[self usernameItem]
        toSectionWithIdentifier:SectionIdentifierPassword];
  }

  // Federated and blocked password forms don't have password value.
  if ([self.password.password length]) {
    self.passwordTextItem = [self passwordItem];
    [model addItem:self.passwordTextItem
        toSectionWithIdentifier:SectionIdentifierPassword];

    if (self.password.isCompromised) {
      [model addSectionWithIdentifier:SectionIdentifierCompromisedInfo];

      if (self.password.changePasswordURL.is_valid()) {
        [model addItem:[self changePasswordItem]
            toSectionWithIdentifier:SectionIdentifierCompromisedInfo];
      }

      [model addItem:[self changePasswordRecommendationItem]
          toSectionWithIdentifier:SectionIdentifierCompromisedInfo];
    }
  }
}

#pragma mark - Items

- (TableViewTextEditItem*)websiteItem {
  TableViewTextEditItem* item =
      [[TableViewTextEditItem alloc] initWithType:ItemTypeWebsite];
  item.textFieldName = l10n_util::GetNSString(IDS_IOS_SHOW_PASSWORD_VIEW_SITE);
  item.textFieldValue = self.password.website;
  item.textFieldEnabled = NO;
  item.hideIcon = YES;
  return item;
}

- (TableViewTextEditItem*)usernameItem {
  TableViewTextEditItem* item =
      [[TableViewTextEditItem alloc] initWithType:ItemTypeUsername];
  item.textFieldName =
      l10n_util::GetNSString(IDS_IOS_SHOW_PASSWORD_VIEW_USERNAME);
  item.textFieldValue = self.password.username;
  item.textFieldEnabled = NO;
  item.hideIcon = YES;
  return item;
}

- (TableViewTextEditItem*)passwordItem {
  TableViewTextEditItem* item =
      [[TableViewTextEditItem alloc] initWithType:ItemTypePassword];
  item.textFieldName =
      l10n_util::GetNSString(IDS_IOS_SHOW_PASSWORD_VIEW_PASSWORD);
  item.textFieldValue = [self isPasswordShown] || self.tableView.editing
                            ? self.password.password
                            : kMaskedPassword;
  item.textFieldEnabled = self.tableView.editing;
  item.hideIcon = !self.tableView.editing;
  item.autoCapitalizationType = UITextAutocapitalizationTypeNone;
  item.keyboardType = UIKeyboardTypeURL;
  item.returnKeyType = UIReturnKeyDone;

  // During editing password is exposed so eye icon shouldn't be shown.
  if (!self.tableView.editing) {
    NSString* image = [self isPasswordShown] ? @"infobar_hide_password_icon"
                                             : @"infobar_reveal_password_icon";
    item.identifyingIcon = [[UIImage imageNamed:image]
        imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    item.identifyingIconEnabled = YES;
  }
  return item;
}

- (TableViewTextItem*)changePasswordItem {
  TableViewTextItem* item =
      [[TableViewTextItem alloc] initWithType:ItemTypeChangePasswordButton];
  item.text = l10n_util::GetNSString(IDS_IOS_CHANGE_COMPROMISED_PASSWORD);
  item.textColor = self.tableView.editing ? UIColor.cr_secondaryLabelColor
                                          : [UIColor colorNamed:kBlueColor];
  item.accessibilityTraits = UIAccessibilityTraitButton;
  return item;
}

- (SettingsImageDetailTextItem*)changePasswordRecommendationItem {
  SettingsImageDetailTextItem* item = [[SettingsImageDetailTextItem alloc]
      initWithType:ItemTypeChangePasswordRecommendation];
  item.detailText =
      l10n_util::GetNSString(IDS_IOS_CHANGE_COMPROMISED_PASSWORD_DESCRIPTION);
  item.image = [self getCompromisedIcon];
  return item;
}

#pragma mark - UITableViewDelegate

- (void)tableView:(UITableView*)tableView
    didSelectRowAtIndexPath:(NSIndexPath*)indexPath {
  TableViewModel* model = self.tableViewModel;
  NSInteger itemType = [model itemTypeForIndexPath:indexPath];
  switch (itemType) {
    case ItemTypeWebsite:
    case ItemTypeUsername:
    case ItemTypeChangePasswordRecommendation:
      break;
    case ItemTypePassword: {
      if (self.tableView.editing) {
        UITableViewCell* cell =
            [self.tableView cellForRowAtIndexPath:indexPath];
        TableViewTextEditCell* textFieldCell =
            base::mac::ObjCCastStrict<TableViewTextEditCell>(cell);
        [textFieldCell.textField becomeFirstResponder];
      }
      break;
    }
    case ItemTypeChangePasswordButton:
      if (!self.tableView.editing) {
        DCHECK(self.commandsDispatcher);
        DCHECK(self.password.changePasswordURL.is_valid());
        OpenNewTabCommand* command = [OpenNewTabCommand
            commandWithURLFromChrome:self.password.changePasswordURL];
        [self.commandsDispatcher closeSettingsUIAndOpenURL:command];
      }
      break;
  }
}

- (UITableViewCellEditingStyle)tableView:(UITableView*)tableView
           editingStyleForRowAtIndexPath:(NSIndexPath*)indexPath {
  return UITableViewCellEditingStyleNone;
}

- (BOOL)tableView:(UITableView*)tableview
    shouldIndentWhileEditingRowAtIndexPath:(NSIndexPath*)indexPath {
  return NO;
}

#pragma mark - UITableViewDataSource

- (UITableViewCell*)tableView:(UITableView*)tableView
        cellForRowAtIndexPath:(NSIndexPath*)indexPath {
  UITableViewCell* cell = [super tableView:tableView
                     cellForRowAtIndexPath:indexPath];

  cell.selectionStyle = UITableViewCellSelectionStyleNone;

  NSInteger itemType = [self.tableViewModel itemTypeForIndexPath:indexPath];
  switch (itemType) {
    case ItemTypePassword: {
      TableViewTextEditCell* textFieldCell =
          base::mac::ObjCCastStrict<TableViewTextEditCell>(cell);
      textFieldCell.textField.delegate = self;
      [textFieldCell.identifyingIconButton
                 addTarget:self
                    action:@selector(didTapShowHideButton:)
          forControlEvents:UIControlEventTouchUpInside];
      return textFieldCell;
    }
    case ItemTypeChangePasswordButton:
      cell.selectionStyle = UITableViewCellSelectionStyleDefault;
      break;
    case ItemTypeWebsite:
    case ItemTypeUsername:
    case ItemTypeChangePasswordRecommendation:
      break;
  }
  return cell;
}

- (BOOL)tableView:(UITableView*)tableView
    canEditRowAtIndexPath:(NSIndexPath*)indexPath {
  NSInteger itemType = [self.tableViewModel itemTypeForIndexPath:indexPath];
  switch (itemType) {
    case ItemTypeWebsite:
    case ItemTypeUsername:
      return NO;
    case ItemTypePassword:
      return YES;
  }
  return NO;
}

#pragma mark - PasswordDetailsConsumer

- (void)setPassword:(PasswordDetails*)password {
  _password = password;
  [self reloadData];
}

#pragma mark - Private

// Called when user tapped Delete button during editing. It means presented
// password should be deleted.
- (void)deleteItems:(NSArray<NSIndexPath*>*)indexPaths {
  // Pass origin only if password is compromised as confirmation message makes
  // sense only in this case.
  if (self.password.isCompromised) {
    [self.handler showPasswordDeleteDialogWithOrigin:self.password.origin];
  } else {
    [self.handler showPasswordDeleteDialogWithOrigin:nil];
  }
}

- (BOOL)shouldHideToolbar {
  return !self.editing;
}

// Applies tint colour and resizes image.
- (UIImage*)getCompromisedIcon {
  UIImage* image = [UIImage imageNamed:@"settings_unsafe_state"];
  UIImage* newImage =
      [image imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
  UIGraphicsBeginImageContextWithOptions(
      CGSizeMake(kWarningIconSize, kWarningIconSize), NO, 0.0);
  [UIColor.cr_secondaryLabelColor set];
  [newImage drawInRect:CGRectMake(0, 0, kWarningIconSize, kWarningIconSize)];
  newImage = UIGraphicsGetImageFromCurrentImageContext();
  UIGraphicsEndImageContext();
  return newImage;
}

// Shows reauthentication dialog if needed. If the reauthentication is
// successful reveals the password.
- (void)attemptToShowPasswordFor:(ReauthenticationReason)reason {
  // If password was already shown (before editing or copying) we don't need to
  // request reauth again.
  if (self.isPasswordShown) {
    [self showPasswordFor:reason];
    return;
  }

  if ([self.reauthModule canAttemptReauth]) {
    __weak __typeof(self) weakSelf = self;
    void (^showPasswordHandler)(ReauthenticationResult) =
        ^(ReauthenticationResult result) {
          PasswordDetailsTableViewController* strongSelf = weakSelf;
          if (!strongSelf)
            return;
          [strongSelf logPasswordSettingsReauthResult:result];

          if (result == ReauthenticationResult::kFailure)
            return;

          [strongSelf showPasswordFor:reason];
        };

    [self.reauthModule
        attemptReauthWithLocalizedReason:[self getLocalizedStringFor:reason]
                    canReusePreviousAuth:YES
                                 handler:showPasswordHandler];
  } else {
    [self.handler showPasscodeDialog];
  }
}

// Reveals password to the user.
- (void)showPasswordFor:(ReauthenticationReason)reason {
  switch (reason) {
    case ReauthenticationReasonShow:
      self.passwordShown = YES;
      self.passwordTextItem.textFieldValue = self.password.password;
      self.passwordTextItem.identifyingIcon =
          [[UIImage imageNamed:@"infobar_hide_password_icon"]
              imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
      [self reconfigureCellsForItems:@[ self.passwordTextItem ]];
      break;
    case ReauthenticationReasonCopy:
      // TODO:(crbug.com/1075494) - Implement copy password functionality.
      break;
    case ReauthenticationReasonEdit:
      // Called super because we want to update only |tableView.editing|.
      [super editButtonPressed];
      [self reloadData];
      break;
  }
  [self logPasswordAccessWith:reason];
}

// Returns localized reason for reauthentication dialog.
- (NSString*)getLocalizedStringFor:(ReauthenticationReason)reason {
  switch (reason) {
    case ReauthenticationReasonShow:
      return l10n_util::GetNSString(
          IDS_IOS_SETTINGS_PASSWORD_REAUTH_REASON_SHOW);
    case ReauthenticationReasonCopy:
      return l10n_util::GetNSString(
          IDS_IOS_SETTINGS_PASSWORD_REAUTH_REASON_COPY);
    case ReauthenticationReasonEdit:
      return l10n_util::GetNSString(
          IDS_IOS_SETTINGS_PASSWORD_REAUTH_REASON_EDIT);
  }
}

- (void)passwordEditingConfirmed {
  self.password.password = self.passwordTextItem.textFieldValue;
  [self.delegate passwordDetailsViewController:self
                        didEditPasswordDetails:self.password];
  [super editButtonPressed];
  [self reloadData];
}

- (BOOL)isItemAtIndexPathTextEditCell:(NSIndexPath*)cellPath {
  NSInteger itemType = [self.tableViewModel itemTypeForIndexPath:cellPath];
  switch (static_cast<ItemType>(itemType)) {
    case ItemTypePassword:
      return YES;
    case ItemTypeWebsite:
    case ItemTypeUsername:
    case ItemTypeChangePasswordButton:
    case ItemTypeChangePasswordRecommendation:
      return NO;
  }
}

#pragma mark - Actions

// Called when the user tapped on the show/hide button near password.
- (void)didTapShowHideButton:(UIButton*)buttonView {
  if (self.isPasswordShown) {
    self.passwordShown = NO;
    self.passwordTextItem.textFieldValue = kMaskedPassword;
    self.passwordTextItem.identifyingIcon =
        [[UIImage imageNamed:@"infobar_reveal_password_icon"]
            imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    [self reconfigureCellsForItems:@[ self.passwordTextItem ]];
  } else {
    [self attemptToShowPasswordFor:ReauthenticationReasonShow];
  }
}

#pragma mark - Metrics

// Logs metrics for the given reauthentication |result| (success, failure or
// skipped).
- (void)logPasswordSettingsReauthResult:(ReauthenticationResult)result {
  switch (result) {
    case ReauthenticationResult::kSuccess:
      LogPasswordSettingsReauthResult(ReauthResult::kSuccess);
      break;
    case ReauthenticationResult::kFailure:
      LogPasswordSettingsReauthResult(ReauthResult::kFailure);
      break;
    case ReauthenticationResult::kSkipped:
      LogPasswordSettingsReauthResult(ReauthResult::kSkipped);
      break;
  }
}

- (void)logPasswordAccessWith:(ReauthenticationReason)reason {
  switch (reason) {
    case ReauthenticationReasonShow:
      UMA_HISTOGRAM_ENUMERATION(
          "PasswordManager.AccessPasswordInSettings",
          password_manager::metrics_util::ACCESS_PASSWORD_VIEWED,
          password_manager::metrics_util::ACCESS_PASSWORD_COUNT);
      break;
    case ReauthenticationReasonCopy:
      UMA_HISTOGRAM_ENUMERATION(
          "PasswordManager.AccessPasswordInSettings",
          password_manager::metrics_util::ACCESS_PASSWORD_COPIED,
          password_manager::metrics_util::ACCESS_PASSWORD_COUNT);
      break;
    case ReauthenticationReasonEdit:
      UMA_HISTOGRAM_ENUMERATION(
          "PasswordManager.AccessPasswordInSettings",
          password_manager::metrics_util::ACCESS_PASSWORD_EDITED,
          password_manager::metrics_util::ACCESS_PASSWORD_COUNT);
      break;
  }
}

@end
