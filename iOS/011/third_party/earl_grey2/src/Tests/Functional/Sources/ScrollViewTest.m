//
// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#import "GREYConfigKey.h"
#import "GREYErrorConstants.h"
#import "GREYConstants.h"
#import "GREYDescription.h"
#import "EarlGrey.h"
#import "GREYHostApplicationDistantObject+ScrollViewTest.h"
#import "BaseIntegrationTest.h"

@interface ScrollViewTest : BaseIntegrationTest
@end

@implementation ScrollViewTest

- (void)setUp {
  [super setUp];
  [self openTestViewNamed:@"Scroll Views"];
}

- (void)testScrollToTopEdge {
  id<GREYMatcher> matcher = grey_allOf(grey_accessibilityLabel(@"Label 2"), grey_interactable(),
                                       grey_sufficientlyVisible(), nil);
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionDown, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_sufficientlyVisible()];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeTop)];
}

- (void)testScrollToBottomEdge {
  [[[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeBottom)]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeBottom)];
}

- (void)testScrollToRightEdge {
  [[[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeRight)]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeRight)];
}

- (void)testScrollToLeftEdge {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeRight)];
  [[[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeLeft)]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeLeft)];
}

- (void)testScrollToLeftEdgeWithCustomStartPoint {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdgeWithStartPoint(kGREYContentEdgeLeft, 0.5, 0.5)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeLeft)];
}

- (void)testScrollToRightEdgeWithCustomStartPoint {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdgeWithStartPoint(kGREYContentEdgeRight, 0.5, 0.5)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeRight)];
}

- (void)testScrollToTopEdgeWithCustomStartPoint {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdgeWithStartPoint(kGREYContentEdgeTop, 0.5, 0.5)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeTop)];
}

- (void)testScrollToBottomEdgeWithCustomStartPoint {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      performAction:grey_scrollToContentEdgeWithStartPoint(kGREYContentEdgeBottom, 0.5, 0.5)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeBottom)];
}

- (void)testScrollToTopWorksWithPositiveInsets {
  // Scroll down.
  id<GREYMatcher> matcher = grey_allOf(grey_accessibilityLabel(@"Label 2"), grey_interactable(),
                                       grey_sufficientlyVisible(), nil);
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionDown, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_sufficientlyVisible()];

  // Add positive insets using this format {top,left,bottom,right}
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      performAction:grey_typeText(@"{100,0,0,0}\n")];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"insets toggle")]
      performAction:grey_turnSwitchOn(YES)];

  // Scroll to top and verify.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeTop)];
}

- (void)testScrollToTopWorksWithNegativeInsets {
  // Scroll down.
  id<GREYMatcher> matcher =
      grey_allOf(grey_accessibilityLabel(@"Label 2"), grey_interactable(), nil);
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionDown, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_sufficientlyVisible()];

  // Add positive insets using this format {top,left,bottom,right}
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      performAction:grey_typeText(@"{-100,0,0,0}\n")];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"insets toggle")]
      performAction:grey_turnSwitchOn(YES)];

  // Scroll to top and verify.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeTop)];
}

- (void)testSearchActionReturnsNilWhenElementIsNotFound {
  id<GREYMatcher> matcher =
      grey_allOf(grey_accessibilityLabel(@"Unobtainium"), grey_interactable(), nil);
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionUp, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_nil()];
}

- (void)testScrollToTopWhenAlreadyAtTheTopWithoutBounce {
  GREYHostApplicationDistantObject *host = GREYHostApplicationDistantObject.sharedInstance;
  id<GREYAction> bounceOff = [host actionForToggleBounces];

  // Verify this test with and without bounce enabled by toggling it.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:bounceOff];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeTop)];
}

- (void)testScrollToTopWhenAlreadyAtTheTopWithBounce {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];

  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_scrolledToContentEdge(kGREYContentEdgeTop)];
}

- (void)testVisibilityOnPartiallyObscuredScrollView {
  if (iOS13_OR_ABOVE()) {
    GREYHostApplicationDistantObject *host = GREYHostApplicationDistantObject.sharedInstance;
    id<GREYAssertion> assertion = [host assertionWithPartiallyVisible];
    [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Bottom Scroll View")]
        assert:assertion];
  }
}

- (void)testInfiniteScroll {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 100)];
}

- (void)testScrollInDirectionCausesExactChangesToContentOffsetInPortraitMode {
  [self ftr_assertScrollInDirectionCausesExactChangesToContentOffset];
}

- (void)testScrollInDirectionCausesExactChangesToContentOffsetInPortraitUpsideDownMode {
  [EarlGrey rotateDeviceToOrientation:UIDeviceOrientationPortraitUpsideDown error:nil];
  [self ftr_assertScrollInDirectionCausesExactChangesToContentOffset];
}

- (void)testScrollInDirectionCausesExactChangesToContentOffsetInLandscapeLeftMode {
  [EarlGrey rotateDeviceToOrientation:UIDeviceOrientationLandscapeLeft error:nil];
  [self ftr_assertScrollInDirectionCausesExactChangesToContentOffset];
}

- (void)testScrollInDirectionCausesExactChangesToContentOffsetInLandscapeRightMode {
  [EarlGrey rotateDeviceToOrientation:UIDeviceOrientationLandscapeRight error:nil];
  [self ftr_assertScrollInDirectionCausesExactChangesToContentOffset];
}

// TODO: Because the action is performed outside the main thread, the synchronization // NOLINT
//       waits until the scrolling stops, where the scroll view's inertia causes itself
//       to move more than needed.
- (void)testScrollInDirectionCausesExactChangesToContentOffsetWithTinyScrollAmounts {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 7)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(0, 7)))];
  // Go right to (6, 7)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionRight, 6)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(6, 7)))];
  // Go up to (6, 4)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionUp, 3)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(6, 4)))];
  // Go left to (3, 4)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionLeft, 3)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(3, 4)))];
}

- (void)testScrollToTopWithZeroXOffset {
  // Scroll down.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 500)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(0, 500)))];
  // Scroll up using grey_scrollToTop(...) and verify scroll offset is back at 0.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(0, 0)))];
}

- (void)testScrollToTopWithNonZeroXOffset {
  // Scroll to (50, 370) as going higher might cause bouncing because of iOS 13+ autoresizing.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 370)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionRight, 50)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(50, 370)))];
  // Scroll up using grey_scrollToContentEdge(...) and verify scroll offset is back at 0.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeTop)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(50, 0)))];
}

- (void)testScrollingBeyondTheContentViewCausesScrollErrors {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 100)];
  NSError *scrollError;
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionUp, 200)
              error:&scrollError];
  GREYAssertEqualObjects(scrollError.domain, kGREYScrollErrorDomain, @"should be equal");
  GREYAssertEqual(scrollError.code, kGREYScrollReachedContentEdge, @"should be equal");
}

- (void)testSetContentOffsetAnimatedYesWaitsForAnimation {
  [self ftr_setContentOffSet:CGPointMake(0, 100) animated:YES];

  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"SquareElementLabel")]
      assertWithMatcher:grey_sufficientlyVisible()];
}

- (void)testSetContentOffsetAnimatedNoDoesNotWaitForAnimation {
  [self ftr_setContentOffSet:CGPointMake(0, 100) animated:NO];

  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"SquareElementLabel")]
      assertWithMatcher:grey_sufficientlyVisible()];
}

- (void)testSetContentOffsetToSameCGPointDoesNotWait {
  [self ftr_setContentOffSet:CGPointZero animated:YES];
}

- (void)testContentSizeSmallerThanViewSize {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Small Content Scroll View")]
      performAction:grey_scrollToContentEdge(kGREYContentEdgeBottom)];
}

/** Verifies that the NSTimer and animation for making the scroll bar disappear is called. */
- (void)testScrollIndicatorRemovalImmediatelyAfterAnAction {
  id<GREYMatcher> infiniteScrollViewMatcher = grey_accessibilityLabel(@"Infinite Scroll View");
  [[EarlGrey selectElementWithMatcher:infiniteScrollViewMatcher]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 99)];
  id<GREYMatcher> axValueMatcher = grey_allOf(grey_ancestor(infiniteScrollViewMatcher),
                                              InfiniteScrollViewIndicatorMatcher(), nil);
  [GREYConfiguration.sharedConfiguration setValue:@(NO)
                                     forConfigKey:kGREYConfigKeySynchronizationEnabled];
  [[EarlGrey selectElementWithMatcher:axValueMatcher] assertWithMatcher:grey_sufficientlyVisible()];
  [GREYConfiguration.sharedConfiguration setValue:@(YES)
                                     forConfigKey:kGREYConfigKeySynchronizationEnabled];

  [[EarlGrey selectElementWithMatcher:infiniteScrollViewMatcher]
      performAction:grey_scrollInDirection(kGREYDirectionUp, 99)];
  [[EarlGrey selectElementWithMatcher:axValueMatcher] assertWithMatcher:grey_notVisible()];
}

/** Scroll Indicators should be tracked post a scroll action being done. */
- (void)testScrollIndicatorRemovalAfterTurningOffSynchronizationAndPerformingAScrollAction {
  id<GREYMatcher> infiniteScrollViewMatcher = grey_accessibilityLabel(@"Infinite Scroll View");
  [[EarlGrey selectElementWithMatcher:infiniteScrollViewMatcher]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 99)];
  [[GREYConfiguration sharedConfiguration] setValue:@(NO)
                                       forConfigKey:kGREYConfigKeySynchronizationEnabled];
  [[EarlGrey selectElementWithMatcher:infiniteScrollViewMatcher]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 99)];
  [[GREYConfiguration sharedConfiguration] setValue:@(YES)
                                       forConfigKey:kGREYConfigKeySynchronizationEnabled];
  id<GREYMatcher> axValueMatcher = grey_allOf(grey_ancestor(infiniteScrollViewMatcher),
                                              InfiniteScrollViewIndicatorMatcher(), nil);
  [[EarlGrey selectElementWithMatcher:axValueMatcher] assertWithMatcher:grey_notVisible()];
}

#pragma mark - Private

/**
 * @return A GREYMatcher showing us the scroll indicator for the Infinite ScrollView.
 */
static id<GREYMatcher> InfiniteScrollViewIndicatorMatcher() {
  return [GREYElementMatcherBlock
      matcherWithMatchesBlock:^BOOL(NSObject *element) {
        return [element.accessibilityLabel containsString:@"Vertical scroll bar"];
      }
      descriptionBlock:^(id<GREYDescription> _Nonnull description) {
        [description appendText:@"Indicator not present"];
      }];
}
// Asserts that the scroll actions work accurately in all four directions by verifying the content
// offset changes caused by them.
- (void)ftr_assertScrollInDirectionCausesExactChangesToContentOffset {
  // Scroll by a fixed amount and verify that the scroll offset has changed by that amount.
  // Go down to (0, 99)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 99)];

  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(0, 99)))];
  // Go right to (77, 99)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionRight, 77)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(77, 99)))];
  // Go up to (77, 44)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionUp, 55)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(77, 44)))];
  // Go left to (33, 44)
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Infinite Scroll View")]
      performAction:grey_scrollInDirection(kGREYDirectionLeft, 44)];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"topTextbox")]
      assertWithMatcher:grey_text(NSStringFromCGPoint(CGPointMake(33, 44)))];
}

// Makes a setContentOffset:animated: call on an element of type UIScrollView.
- (void)ftr_setContentOffSet:(CGPoint)offset animated:(BOOL)animated {
  GREYHostApplicationDistantObject *host = GREYHostApplicationDistantObject.sharedInstance;
  id<GREYAction> action = [host actionForSetScrollViewContentOffSet:offset animated:animated];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      performAction:action];
}

@end
