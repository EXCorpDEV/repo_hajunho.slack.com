//
// Copyright 2020 Google Inc.
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
#import "GREYMatchersShorthand.h"
#import "GREYConfigKey.h"
#import "GREYError.h"
#import "GREYErrorConstants.h"
#import "GREYConstants.h"
#import "GREYWaitFunctions.h"
#import "EarlGrey.h"
#import "GREYHostApplicationDistantObject+ErrorHandlingTest.h"
#import "BaseIntegrationTest.h"
#import "FailureHandler.h"

#pragma mark - Failure Handler

/**
 * Failure handler used for testing the console output of failures
 */
@interface FailureFormatTestingFailureHandler : NSObject <GREYFailureHandler>

/** The filename where the failure is located at. */
@property NSString *fileName;

/** The line number where the failure is located at. */
@property(assign) NSUInteger lineNumber;

/** Exception to handle the failure for*/
@property GREYFrameworkException *exception;

/** Details for the exception. */
@property NSString *details;
@end

@implementation FailureFormatTestingFailureHandler

- (void)handleException:(GREYFrameworkException *)exception details:(NSString *)details {
  self.exception = exception;
  self.details = details;
}

- (void)setInvocationFile:(NSString *)fileName andInvocationLine:(NSUInteger)lineNumber {
  self.fileName = fileName;
  self.lineNumber = lineNumber;
}

@end

#pragma mark - FailureFormattingTest

/**
 * Verifies that the user-facing console output follows expectations.
 */
@interface FailureFormattingTest : BaseIntegrationTest
@end

@implementation FailureFormattingTest {
  /** Custom failure handler for checking the formatting. */
  FailureFormatTestingFailureHandler *_handler;
  /** The original failure handler. */
  id<GREYFailureHandler> _originalHandler;
}

- (void)setUp {
  [super setUp];
  _originalHandler = [NSThread mainThread].threadDictionary[GREYFailureHandlerKey];
  _handler = [[FailureFormatTestingFailureHandler alloc] init];
  [NSThread mainThread].threadDictionary[GREYFailureHandlerKey] = _handler;
}

- (void)tearDown {
  [NSThread mainThread].threadDictionary[GREYFailureHandlerKey] = _originalHandler;
  [super tearDown];
}

/** Tests localized description returns the whole error description not just the error reason. */
- (void)testLocalizedDescription {
  NSError *error;
  [[EarlGrey selectElementWithMatcher:grey_kindOfClass([UITableViewCell class])]
      performAction:grey_tap()
              error:&error];
  XCTAssertTrue([error.localizedDescription
      containsString:@"Multiple elements were matched. Please use selection matchers to narrow the "
                     @"selection down to a single element."]);
  XCTAssertTrue([error.localizedDescription containsString:@"Elements Matched:"]);
  XCTAssertTrue([error.localizedDescription containsString:@"UI Hierarchy (Back to front):"]);
}

/**
 * Checks the formatting of logs for an element not found error for an assertion without a search
 * action failure.
 */
- (void)testNotFoundAssertionErrorDescription {
  [self openTestViewNamed:@"Animations"];
  [[EarlGrey selectElementWithMatcher:grey_text(@"Basic Views")] assertWithMatcher:grey_notNil()];

  NSString *expectedDetails = @"Interaction cannot continue because the desired element was not "
                              @"found.\n"
                              @"\n"
                              @"Check if the element exists in the UI hierarchy printed below. If "
                              @"it exists, adjust the matcher so that it accurately matches "
                              @"the element.\n"
                              @"\n"
                              @"Element Matcher:\n"
                              @"((kindOfClass('UILabel') || kindOfClass('UITextField') || "
                              @"kindOfClass('UITextView')) && hasText('Basic Views'))\n"
                              @"\n"
                              @"Failed Assertion:\n"
                              @"assertWithMatcher:isNotNil";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

/**
 * Ensures that an assertion define prints the right description.
 */
- (void)testAssertionDefinesContainHierarchyAndScreenshots {
  NSString *assertDescription = @"Assertion Description";
  GREYAssertTrue(NO, assertDescription);

  XCTAssertTrue([_handler.details containsString:assertDescription],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                assertDescription, _handler.details);
}

/**
 * Checks the formatting of logs for an element not found error for an assertion with a search
 * action failure.
 */
- (void)testSearchNotFoundAssertionErrorDescription {
  [self openTestViewNamed:@"Scroll Views"];
  id<GREYMatcher> matcher = grey_allOf(grey_accessibilityLabel(@"Label 2"), grey_interactable(),
                                       grey_sufficientlyVisible(), nil);
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionDown, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Invalid Scroll View")]
      assertWithMatcher:grey_sufficientlyVisible()];

  NSString *expectedDetails =
      @"Search action failed. Look at the underlying error.\n"
      @"\n"
      @"Element Matcher:\n"
      @"(((respondsToSelector(isAccessibilityElement) && "
      @"isAccessibilityElement) && accessibilityLabel('Label 2')) && "
      @"interactable Point:{nan, nan} && sufficientlyVisible(Expected: "
      @"0.750000, Actual: 0.000000))\n"
      @"\n"
      @"Failed Assertion:\n"
      @"assertWithMatcher:sufficientlyVisible(Expected: 0.750000, Actual: 0.000000)\n"
      @"\n"
      @"Search Action:";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

/**
 * Checks the formatting of logs for an element not found error for an action without a search
 * action failure.
 */
- (void)testSearchNotFoundActionErrorDescription {
  [self openTestViewNamed:@"Scroll Views"];
  id<GREYMatcher> matcher = grey_allOf(grey_accessibilityLabel(@"Label 2"), grey_interactable(),
                                       grey_sufficientlyVisible(), nil);
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionDown, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Invalid Scroll View")]
      performAction:grey_tap()];

  NSString *expectedDetails = @"Search action failed. Look at the underlying error.\n"
                              @"\n"
                              @"Element Matcher:\n"
                              @"(((respondsToSelector(isAccessibilityElement) && "
                              @"isAccessibilityElement) && accessibilityLabel('Label 2')) && "
                              @"interactable Point:{nan, nan} && sufficientlyVisible(Expected: "
                              @"0.750000, Actual: 0.000000))\n"
                              @"\n"
                              @"Failed Action:\nTap\n"
                              @"\n"
                              @"Search Action:";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

/**
 * Checks the formatting of logs for an element not found error for an action without a search
 * action failure.
 */
- (void)testNotFoundActionErrorDescription {
  CFTimeInterval originalInteractionTimeout =
      GREY_CONFIG_DOUBLE(kGREYConfigKeyInteractionTimeoutDuration);
  [[GREYConfiguration sharedConfiguration] setValue:@(1)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  NSString *jsStringAboveTimeout =
      @"start = new Date().getTime(); while (new Date().getTime() < start + 3000);";
  // JS action timeout greater than the threshold.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      performAction:grey_javaScriptExecution(jsStringAboveTimeout, nil)];
  [[GREYConfiguration sharedConfiguration] setValue:@(originalInteractionTimeout)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  NSString *expectedDetails = @"Interaction cannot continue because the "
                              @"desired element was not found.\n"
                              @"\n"
                              @"Check if the element exists in the UI hierarchy printed below. If "
                              @"it exists, adjust the matcher so that it accurately matches "
                              @"the element.\n"
                              @"\n"
                              @"Element Matcher:\n"
                              @"(respondsToSelector(accessibilityIdentifier) && "
                              @"accessibilityID('TestWKWebView'))\n"
                              @"\n"
                              @"Failed Action:\n"
                              @"Execute JavaScript";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

/**
 * Checks the formatting of logs for timeout error.
 */
- (void)testTimeoutErrorDescription {
  [self openTestViewNamed:@"Scroll Views"];
  id<GREYMatcher> matcher = grey_allOf(grey_accessibilityLabel(@"Label 2"), grey_interactable(),
                                       grey_sufficientlyVisible(), nil);
  [[GREYConfiguration sharedConfiguration] setValue:@(1)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  [[[EarlGrey selectElementWithMatcher:matcher]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionDown, 50)
      onElementWithMatcher:grey_accessibilityLabel(@"Upper Scroll View")]
      assertWithMatcher:grey_sufficientlyVisible()];
  NSString *expectedDetails =
      @"Interaction timed out after 1 seconds while searching "
      @"for element.\n"
      @"\n"
      @"Increase timeout for matching element.\n"
      @"\n"
      @"Element Matcher:\n"
      @"(((respondsToSelector(isAccessibilityElement) && "
      @"isAccessibilityElement) && accessibilityLabel('Label 2')) && "
      @"interactable Point:{nan, nan} && sufficientlyVisible(Expected: "
      @"0.750000, Actual: 0.000000))\n"
      @"\n"
      @"Failed Assertion:\n"
      @"assertWithMatcher:sufficientlyVisible(Expected: 0.750000, Actual: 0.000000)\n"
      @"\n"
      @"UI Hierarchy";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

/**
 * Checks the formatting for a type interaction failing.
 */
- (void)testActionInteractionErrorDescription {
  [[EarlGrey selectElementWithMatcher:grey_text(@"Basic Views")] performAction:grey_tap()];
  [[EarlGrey selectElementWithMatcher:grey_text(@"Tab 2")] performAction:grey_tap()];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"foo")]
      performAction:grey_typeText(@"")];

  NSString *expectedDetails =
      @"Failed to type because the provided string was empty.\n"
      @"\n"
      @"Element Matcher:\n"
      @"(respondsToSelector(accessibilityIdentifier) && accessibilityID('foo'))\n"
      @"\n"
      @"UI Hierarchy";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

- (void)testMultipleMatchedErrorDescription {
  [[EarlGrey selectElementWithMatcher:grey_kindOfClass([UITableViewCell class])]
      performAction:grey_tap()
              error:nil];

  NSString *expectedDetails = @"Multiple elements were matched. Please use selection matchers "
                              @"to narrow the selection down to a single element.\n"
                              @"\n"
                              @"Create a more specific matcher to uniquely match the element. "
                              @"In general, prefer using accessibility ID before accessibility "
                              @"label or other attributes.\n"
                              @"Use atIndex: to select from one of the matched elements. "
                              @"Keep in mind when using atIndex: that the order in which "
                              @"elements are arranged may change, making your test brittle.\n"
                              @"\n"
                              @"Element Matcher:\n"
                              @"kindOfClass('UITableViewCell')\n"
                              @"\n"
                              @"Failed Action:\n"
                              @"Tap\n"
                              @"\n"
                              @"Elements Matched:\n"
                              @"\n"
                              @"\t1.";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
  XCTAssertTrue([_handler.details containsString:@"UI Hierarchy"],
                @"\"UI Hierarchy\" does not appear in the actual exception details:\n\n"
                @"========== exception details ==========\n%@",
                _handler.details);
}

- (void)testConstraintsFailureErrorDescription {
  [[EarlGrey selectElementWithMatcher:grey_text(@"Basic Views")] performAction:grey_tap()];
  [[EarlGrey selectElementWithMatcher:grey_buttonTitle(@"Disabled")]
      performAction:grey_scrollInDirection(kGREYDirectionUp, 20)
              error:nil];
  NSString *expectedDetails1 = @"Cannot perform action due to constraint(s) failure.\n"
                               @"\n"
                               @"Adjust element properties so that it matches the failed "
                               @"constraint(s).\n"
                               @"\n"
                               @"Element Matcher:\n"
                               @"(kindOfClass('UIButton') && buttonTitle('Disabled'))\n"
                               @"\n"
                               @"Failed Constraint(s):\n"
                               @"kindOfClass('UIScrollView')kindOfClass('WKWebView'), \n"
                               @"\n"
                               @"Element Description:\n"
                               @"<UIButton:";
  NSString *expectedDetails2 = @"Failed Action:\n"
                               @"Scroll Up for 20\n"
                               @"\n"
                               @"UI Hierarchy";
  XCTAssertTrue([_handler.details containsString:expectedDetails1],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails1, _handler.details);

  XCTAssertTrue([_handler.details containsString:expectedDetails2],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails2, _handler.details);
}

/**
 * Checks the formatting for an assertion failure.
 */
- (void)testAssertionInteractionErrorDescription {
  [[EarlGrey selectElementWithMatcher:grey_keyWindow()] assertWithMatcher:grey_nil()];
  NSString *expectedDetailsTillElement = @"Element does not meet assertion criteria:\nisNil\n\n"
                                          "Element:\n<UIWindow:";
  NSString *expectedDetailsForMatcher = @"\n\nMismatch:\nisNil\n\nElement Matcher:\n"
                                        @"(kindOfClass('UIWindow') && keyWindow)\n\nUI Hierarchy";
  XCTAssertTrue([_handler.details containsString:expectedDetailsTillElement],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetailsTillElement, _handler.details);
  XCTAssertTrue([_handler.details containsString:expectedDetailsForMatcher],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetailsForMatcher, _handler.details);
}

/**
 * Checks the formatting for the failure seen when the index passed in is more than the total number
 * of elements.
 */
- (void)testMatchedElementOutOfBoundsDescription {
  [self openTestViewNamed:@"Typing Views"];
  [[[EarlGrey selectElementWithMatcher:grey_kindOfClass([UITextField class])] atIndex:999]
      assertWithMatcher:grey_notNil()];

  NSString *expectedDetails =
      @"3 elements were matched, but element at index 999 was requested.\n\n"
      @"Please use an element index from 0 to 2.\n\n"
      @"Element Matcher:\n"
      @"kindOfClass('UITextField')";
  XCTAssertTrue([_handler.details containsString:expectedDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetails, _handler.details);
}

/**
 * Checks that the element matcher and UI hierarchy are present in the WKWebView error.
 */
- (void)testWKWebViewFormatting {
  [self openTestViewNamed:@"WKWebView"];
  id<GREYAction> jsAction = grey_javaScriptExecution(@"var foo; foo.bar();", nil);
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")] performAction:jsAction
                                                                                      error:nil];
  NSString *jsErrorDetails = @"TypeError: undefined is not an object (evaluating 'foo.bar')";
  XCTAssertTrue([_handler.details containsString:jsErrorDetails],
                @"Exception:%@ doesn't have the JS Error Details:%@", _handler.details,
                jsErrorDetails);
  XCTAssertTrue([_handler.details containsString:@"Element Matcher:"],
                @"Exception:%@ doesn't have the Element Matcher:", _handler.details);
  XCTAssertTrue([_handler.details containsString:@"UI Hierarchy"],
                @"Exception:%@ doesn't have the UI Hierarchy:", _handler.details);
  XCTAssertTrue([_handler.details containsString:@"Failed Action:\nExecute JavaScript\n"],
                @"Exception:%@ doesn't have the JavaScript Action Name:", _handler.details);
}

/**
 * Ensures that the matcher, interaction and idling resources are present with an interaction
 * timeout.
 */
- (void)testTimeoutForSynchronizationFailure {
  [self openTestViewNamed:@"Animations"];
  [[GREYConfiguration sharedConfiguration] setValue:@(1)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"AnimationControl")]
      performAction:grey_tap()];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"AnimationStatus")]
      assertWithMatcher:grey_text(@"Paused")];
  NSString *idlingResourceInfo = @"The following idling resources are busy.\n\n1.";
  XCTAssertTrue([_handler.details containsString:idlingResourceInfo],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                idlingResourceInfo, _handler.details);
  XCTAssertTrue([_handler.details containsString:@"Element Matcher:"],
                @"Details: %@ does not contain the Element Matcher", _handler.details);
  XCTAssertTrue([_handler.details containsString:@"Failed Assertion:"],
                @"Details: %@ does not contain the Failed Assertion", _handler.details);
}

/** Ensures that the right description is printed when a synthetic event like rotation fails. */
- (void)testSyntheticEventTimeout {
  [[GREYConfiguration sharedConfiguration] setValue:@(0.0)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  [[GREYHostApplicationDistantObject sharedInstance] induceNonTactileActionTimeoutInTheApp];
  [EarlGrey rotateDeviceToOrientation:UIDeviceOrientationPortrait error:nil];
  NSString *exceptionDetails = @"Application did not idle before rotating.\n\n"
                               @"The following idling resources are busy.\n\n1. ";
  XCTAssertTrue([_handler.details containsString:exceptionDetails],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                exceptionDetails, _handler.details);
  // Ensure that the application has idled.
  GREYWaitForAppToIdle(@"Wait for app to idle");
}

/* Ensures that the search action error prints out both wrapped error and underlying error. */
- (void)testSearchActionConstraints {
  [[[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"Invalid")]
         usingSearchAction:grey_scrollInDirection(kGREYDirectionRight, 100)
      onElementWithMatcher:grey_keyWindow()] assertWithMatcher:grey_notNil() error:nil];
  NSString *expectedDetailWrappedError =
      @"Search action failed. Look at the underlying error.\n"
      @"\n"
      @"Element Matcher:\n"
      @"(respondsToSelector(accessibilityIdentifier) && accessibilityID('Invalid'))\n"
      @"\n"
      @"Failed Assertion:\n"
      @"assertWithMatcher:isNotNil\n"
      @"\n"
      @"Search Action:";
  NSString *expectedDetailUnderlyingError =
      @"Cannot perform action due to constraint(s) failure.\n"
      @"\n"
      @"Adjust element properties so that it matches the failed constraint(s).\n"
      @"\n"
      @"Element Matcher:\n"
      @"(kindOfClass('UIWindow') && keyWindow)\n"
      @"\n"
      @"Failed Constraint(s):\n"
      @"kindOfClass('UIScrollView')kindOfClass('WKWebView'),";
  XCTAssertTrue([_handler.details containsString:expectedDetailWrappedError],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetailWrappedError, _handler.details);
  XCTAssertTrue([_handler.details containsString:@"*********** Underlying Error ***********"]);
  XCTAssertTrue([_handler.details containsString:expectedDetailUnderlyingError],
                @"Expected info does not appear in the actual exception details:\n\n"
                @"========== expected info ===========\n%@\n\n"
                @"========== actual exception details ==========\n%@",
                expectedDetailUnderlyingError, _handler.details);
}

@end
