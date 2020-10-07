// Copyright 2019-present the Material Components for iOS authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "MaterialSnapshot.h"

#import <UIKit/UIKit.h>

#import "MaterialAvailability.h"
#import "MaterialButtons.h"
#import "MaterialButtons+Theming.h"
#import "UIView+MDCSnapshot.h"
#import "MaterialContainerScheme.h"

/** A tests fake class of MDCButton. */
@interface MDCButtonSnapshotTestsFakeButton : MDCButton

/** Allows overriding @c traitCollection for testing. */
@property(nonatomic, strong) UITraitCollection *traitCollectionOverride;

@end

@implementation MDCButtonSnapshotTestsFakeButton

- (UITraitCollection *)traitCollection {
  return self.traitCollectionOverride ?: [super traitCollection];
}

@end

/** General snapshot tests for @c MDCButton. */
@interface MDCButtonSnapshotTests : MDCSnapshotTestCase

@end

@implementation MDCButtonSnapshotTests

- (void)setUp {
  [super setUp];

  // Uncomment below to recreate all the goldens (or add the following line to the specific
  // test you wish to recreate the golden for).
  //  self.recordMode = YES;
}

- (void)generateSnapshotAndVerifyForView:(UIView *)view {
  [view sizeToFit];

  UIView *snapshotView = [view mdc_addToBackgroundView];
  [self snapshotVerifyView:snapshotView];
}

- (void)testPreferredFontForAXXXLContentSizeCategory {
  if (@available(iOS 11.0, *)) {
    // Given
    MDCButtonSnapshotTestsFakeButton *button = [[MDCButtonSnapshotTestsFakeButton alloc] init];
    [button applyContainedThemeWithScheme:[[MDCContainerScheme alloc] init]];
    UITraitCollection *xsTraitCollection = [UITraitCollection
        traitCollectionWithPreferredContentSizeCategory:UIContentSizeCategoryExtraSmall];
    UIFont *originalFont = [UIFont preferredFontForTextStyle:UIFontTextStyleBody
                               compatibleWithTraitCollection:xsTraitCollection];
    button.traitCollectionOverride = xsTraitCollection;
    UITraitCollection *aXXXLTraitCollection =
        [UITraitCollection traitCollectionWithPreferredContentSizeCategory:
                               UIContentSizeCategoryAccessibilityExtraExtraExtraLarge];
    [button setTitle:@"Title" forState:UIControlStateNormal];
    button.titleLabel.font = originalFont;
    button.titleLabel.adjustsFontForContentSizeCategory = YES;

    // When
    button.enableTitleFontForState = NO;
    button.traitCollectionOverride = aXXXLTraitCollection;
    // Force the Dynamic Type system to update the button's font.
    [button drawViewHierarchyInRect:button.bounds afterScreenUpdates:YES];

    // Then
    [self generateSnapshotAndVerifyForView:button];
  }
}

- (void)testPreferredFontForXSContentSizeCategory {
  if (@available(iOS 11.0, *)) {
    // Given
    MDCButtonSnapshotTestsFakeButton *button = [[MDCButtonSnapshotTestsFakeButton alloc] init];
    [button applyContainedThemeWithScheme:[[MDCContainerScheme alloc] init]];
    UITraitCollection *aXXXLTraitCollection =
        [UITraitCollection traitCollectionWithPreferredContentSizeCategory:
                               UIContentSizeCategoryAccessibilityExtraExtraExtraLarge];
    UIFont *originalFont = [UIFont preferredFontForTextStyle:UIFontTextStyleBody
                               compatibleWithTraitCollection:aXXXLTraitCollection];
    button.traitCollectionOverride = aXXXLTraitCollection;
    [button setTitle:@"Title" forState:UIControlStateNormal];
    button.titleLabel.font = originalFont;
    button.titleLabel.adjustsFontForContentSizeCategory = YES;

    // When

    button.enableTitleFontForState = NO;
    UITraitCollection *xsTraitCollection = [UITraitCollection
        traitCollectionWithPreferredContentSizeCategory:UIContentSizeCategoryExtraSmall];

    button.traitCollectionOverride = xsTraitCollection;
    // Force the Dynamic Type system to update the button's font.
    [button drawViewHierarchyInRect:button.bounds afterScreenUpdates:YES];

    // Then
    [self generateSnapshotAndVerifyForView:button];
  }
}

- (void)testButtonSupportsDynamicColorScheme {
#if MDC_AVAILABLE_SDK_IOS(13_0)
  if (@available(iOS 13.0, *)) {
    // Given
    MDCButtonSnapshotTestsFakeButton *button = [[MDCButtonSnapshotTestsFakeButton alloc] init];
    [button setTitle:@"Title" forState:UIControlStateNormal];
    MDCContainerScheme *containerScheme = [[MDCContainerScheme alloc] init];
    containerScheme.colorScheme =
        [[MDCSemanticColorScheme alloc] initWithDefaults:MDCColorSchemeDefaultsMaterial201907];
    [button applyContainedThemeWithScheme:containerScheme];

    // When
    UITraitCollection *darkModeTraitCollection =
        [UITraitCollection traitCollectionWithUserInterfaceStyle:UIUserInterfaceStyleDark];
    button.traitCollectionOverride = darkModeTraitCollection;
    [button sizeToFit];

    // Then
    UIView *snapshotView = [button mdc_addToBackgroundView];
    [self snapshotVerifyViewForIOS13:snapshotView];
  }
#endif  // MDC_AVAILABLE_SDK_IOS(13_0)
}

- (void)testButtonWithCustomFrameWhenCenterVisibleArea {
  // Given
  MDCButtonSnapshotTestsFakeButton *button = [[MDCButtonSnapshotTestsFakeButton alloc] init];
  [button applyContainedThemeWithScheme:[[MDCContainerScheme alloc] init]];
  [button setTitle:@"Title" forState:UIControlStateNormal];

  // When
  button.centerVisibleArea = YES;
  button.frame = CGRectMake(0, 0, 100, 100);
  [button layoutIfNeeded];

  // Then
  UIView *snapshotView = [button mdc_addToBackgroundView];
  [self snapshotVerifyView:snapshotView];
}

- (void)testButtonWithCustomCornerRadiusAndCustomFrameWhenCenterVisibleArea {
  // Given
  MDCButtonSnapshotTestsFakeButton *button = [[MDCButtonSnapshotTestsFakeButton alloc] init];
  [button applyContainedThemeWithScheme:[[MDCContainerScheme alloc] init]];
  [button setTitle:@"Title" forState:UIControlStateNormal];

  // When
  button.centerVisibleArea = YES;
  button.layer.cornerRadius = 10;
  button.frame = CGRectMake(0, 0, 100, 100);
  [button layoutIfNeeded];

  // Then
  UIView *snapshotView = [button mdc_addToBackgroundView];
  [self snapshotVerifyView:snapshotView];
}

@end
