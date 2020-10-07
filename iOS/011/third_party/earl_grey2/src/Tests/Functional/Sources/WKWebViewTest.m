//
// Copyright 2018 Google Inc.
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

#import <WebKit/WebKit.h>

#import "BaseIntegrationTest.h"

@interface WKWebViewTest : BaseIntegrationTest
@end

@implementation WKWebViewTest

- (void)setUp {
  [super setUp];
  [self openTestViewNamed:@"WKWebView"];
}

/** Tests scrolling down on a web view. */
- (void)testScrollingWKWebViewWithEarlGrey {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"loadHTMLString")]
      performAction:grey_tap()];
  [self waitForWebViewToLoad];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      performAction:grey_scrollInDirection(kGREYDirectionDown, 20)];
}

/** Tests scrolling to the bottom edge on a web view. */
- (void)testScrollingToContentEdgeWithWKWebViewWithEarlGrey {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityLabel(@"loadHTMLString")]
      performAction:grey_tap()];
  [self waitForWebViewToLoad];

  id<GREYInteraction> webViewInteraction =
      [EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")];
  [webViewInteraction performAction:grey_scrollToContentEdge(kGREYContentEdgeBottom)];

  // Check if the scroll view reached the bottom.
  GREYAssertionBlock *scrollToBottomAssertion =
      [GREYAssertionBlock assertionWithName:@"Did scroll to bottom"
                    assertionBlockWithError:^BOOL(WKWebView *webView, NSError *__strong *error) {
                      UIScrollView *scrollView = webView.scrollView;
                      return scrollView.contentOffset.y + scrollView.frame.size.height >=
                             scrollView.contentSize.height;
                    }];
  [webViewInteraction assert:scrollToBottomAssertion];
}

- (void)testNavigationToWKWebViewTestController {
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      assertWithMatcher:grey_notNil()];
  [[EarlGrey selectElementWithMatcher:grey_buttonTitle(@"Local")] performAction:grey_tap()];
  [[EarlGrey selectElementWithMatcher:grey_kindOfClassName(@"WKScrollView")]
      assertWithMatcher:grey_notNil()];
}

/**
 * Tests executing invalid JavaScript to verify the JavaScript error is propagated to the NSError
 * description.
 */
- (void)testJavascriptEvaluationWithJavascriptError {
  NSError *error = nil;
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      performAction:grey_javaScriptExecution(@"var foo; foo.bar();", nil)
              error:&error];
  XCTAssertEqual(error.code, kGREYWKWebViewInteractionFailedErrorCode);
  NSString *errorString = @"TypeError: undefined is not an object (evaluating 'foo.bar')";
  XCTAssertNotEqual([error.localizedDescription rangeOfString:errorString].location, NSNotFound);
}

- (void)testJavascriptEvaluationWithAReturnValue {
  EDORemoteVariable<NSString *> *javaScriptResult = [[EDORemoteVariable alloc] init];
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      performAction:grey_javaScriptExecution(@"var foo = 1; foo + 1;", javaScriptResult)];
  XCTAssertEqualObjects(javaScriptResult.object, @"2");
}

- (void)testJavascriptEvaluationWithATimeoutAboveTheThreshold {
  NSError *error = nil;
  CFTimeInterval originalInteractionTimeout =
      GREY_CONFIG_DOUBLE(kGREYConfigKeyInteractionTimeoutDuration);
  [[GREYConfiguration sharedConfiguration] setValue:@(1)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  NSString *jsStringAboveTimeout =
      @"start = new Date().getTime(); while (new Date().getTime() < start + 3000);";
  // JS action timeout greater than the threshold.
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      performAction:grey_javaScriptExecution(jsStringAboveTimeout, nil)
              error:&error];
  XCTAssertEqual(error.code, kGREYWKWebViewInteractionFailedErrorCode);
  NSString *timeoutErrorString = @"Interaction with WKWebView failed because of timeout";
  XCTAssertNotEqual([error.localizedDescription rangeOfString:timeoutErrorString].location,
                    NSNotFound);
  [[GREYConfiguration sharedConfiguration] setValue:@(originalInteractionTimeout)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
}

- (void)testJavascriptEvaluationWithATimeoutBelowTheThreshold {
  NSError *error = nil;
  CFTimeInterval originalInteractionTimeout =
      GREY_CONFIG_DOUBLE(kGREYConfigKeyInteractionTimeoutDuration);
  [[GREYConfiguration sharedConfiguration] setValue:@(5)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
  EDORemoteVariable<NSString *> *javaScriptResult = [[EDORemoteVariable alloc] init];
  NSString *jsStringEqualTimeout =
      @"start = new Date().getTime(); while (new Date().getTime() < start + 200);"
      @"end = new Date().getTime(); end - start";
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")]
      performAction:grey_javaScriptExecution(jsStringEqualTimeout, javaScriptResult)
              error:&error];
  XCTAssertNil(error, @"Error for a valid call is nil");
  XCTAssertGreaterThanOrEqual([javaScriptResult.object intValue], 200);
  [[GREYConfiguration sharedConfiguration] setValue:@(originalInteractionTimeout)
                                       forConfigKey:kGREYConfigKeyInteractionTimeoutDuration];
}

- (void)testJavascriptExecutionError {
  EDORemoteVariable<NSString *> *javaScriptResult = [[EDORemoteVariable alloc] init];
  NSError *error;
  id<GREYAction> jsAction =
      grey_javaScriptExecution(@"document.body.getElementsByTagName(\"*\");", javaScriptResult);
  [[EarlGrey selectElementWithMatcher:grey_accessibilityID(@"TestWKWebView")] performAction:jsAction
                                                                                      error:&error];
  XCTAssertEqualObjects(error.domain, kGREYInteractionErrorDomain);
  XCTAssertEqual(error.code, kGREYWKWebViewInteractionFailedErrorCode);
}

#pragma mark - Private

/** Waits for the web view contents to load. */
- (void)waitForWebViewToLoad {
  // TODO(b/145806611): Remove the delay after adding idling resource for WKWebView.
  // Use XCUITest to ensure that the page has loaded.
  XCUIApplication *application = [[XCUIApplication alloc] init];
#if (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 120000)
  XCTAssertTrue([[application.links firstMatch] waitForExistenceWithTimeout:4.0]);
#else
  XCTAssertTrue([[application.links elementBoundByIndex:0] waitForExistenceWithTimeout:4.0]);
#endif
}

@end
