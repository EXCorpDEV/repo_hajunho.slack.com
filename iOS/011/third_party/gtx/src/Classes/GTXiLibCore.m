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

#import "GTXiLibCore.h"

#import "GTXAssertions.h"
#import "GTXChecking.h"
#import "GTXChecksCollection.h"
#import "GTXLogging.h"
#import "GTXPluginXCTestCase.h"
#import "GTXToolKit.h"
#import "NSError+GTXAdditions.h"

#pragma mark - Global definitions.

NSString *const gtxTestCaseDidBeginNotification = @"gtxTestCaseDidBeginNotification";
NSString *const gtxTestCaseDidEndNotification = @"gtxTestCaseDidEndNotification";
NSString *const gtxTestClassUserInfoKey = @"gtxTestClassUserInfoKey";
NSString *const gtxTestInvocationUserInfoKey = @"gtxTestInvocationUserInfoKey";
NSString *const gtxTestInteractionDidBeginNotification = @"gtxTestInteractionDidBeginNotification";
NSString *const gtxTestInteractionDidEndNotification = @"gtxTestInteractionDidEndNotification";

@interface GTXInstallOptions : NSObject

@property (nonatomic, strong) NSArray *checks;
@property (nonatomic, strong) NSArray *elementBlacklist;
@property (nonatomic, strong) GTXTestSuite *suite;

@end

@implementation GTXInstallOptions
@end

/**
 The GTXToolKit instance used to handle accessibility checking in GTXiLib class.
 */
static GTXToolKit *gToolkit;

/**
 The an array of installation options specified by the user so far.
 */
static NSMutableArray *gIntsallOptions;

/**
 The pointer to current options being used by GTXiLib.
 */
static GTXInstallOptions *gCurrentOptions;

/**
 The failure handler block.
 */
static GTXiLibFailureHandler gFailureHandler;

/**
 Boolean that indicates if GTXiLib has detected an on-going interaction.
 */
static BOOL gIsInInteraction;

/**
 Boolean that indicates if GTXiLib has detected a test case tearDown.
 */
static BOOL gIsInTearDown;


#pragma mark - Implementation

@implementation GTXiLib

+ (void)installOnTestSuite:(GTXTestSuite *)suite
                    checks:(NSArray<id<GTXChecking>> *)checks
         elementBlacklists:(NSArray<id<GTXBlacklisting>> *)blacklists {
  [GTXPluginXCTestCase installPlugin];
  if (!gIntsallOptions) {
    gIntsallOptions = [[NSMutableArray alloc] init];
  }

  GTXInstallOptions *options = [[GTXInstallOptions alloc] init];
  options.checks = checks;
  options.elementBlacklist = blacklists;
  options.suite = suite;

  // Assert that this suite has no test cases also specified in other install calls.
  for (GTXInstallOptions *existing in gIntsallOptions) {
    GTXTestSuite *intersection = [existing.suite intersection:suite];
    NSAssert(intersection.tests.count == 0,
             @"Error! Attempting to install GTXChecks multiple times on the same test cases: %@",
             intersection);
    (void)intersection; // Ensures 'intersection' is marked as used even if NSAssert is removed.
  }
  [gIntsallOptions addObject:options];

  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(_testCaseDidBegin:)
                                                 name:gtxTestCaseDidBeginNotification
                                               object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(_testCaseDidTearDown:)
                                                 name:gtxTestCaseDidEndNotification
                                               object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(_testInteractionDidBegin:)
                                                 name:gtxTestInteractionDidBeginNotification
                                               object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(_testInteractionDidFinish:)
                                                 name:gtxTestInteractionDidEndNotification
                                               object:nil];
  });
}

+ (void)setFailureHandler:(GTXiLibFailureHandler)handler {
  gFailureHandler = handler;
}

+ (GTXiLibFailureHandler)failureHandler {
  return gFailureHandler ?: ^(NSError *error) {
    NSString *formattedError =
        [NSString stringWithFormat:@"\n\n%@\n\n",
                                   error.localizedDescription];
    [[NSAssertionHandler currentHandler] handleFailureInMethod:_cmd
                                                        object:self
                                                          file:@(__FILE__)
                                                    lineNumber:__LINE__
                                                   description:@"%@", formattedError];
  };
}

+ (id<GTXChecking>)checkWithName:(NSString *)name block:(GTXCheckHandlerBlock)block {
  return [GTXToolKit checkWithName:name block:block];
}

#pragma mark - Private

/**
 Executes the currently installed checks on the given element. In case of failures, the failure
 handler is invoked.

 @param element The element on which the checks need to be executed.
 @return @c NO if any of the checks failed, @c YES otherwise.
 */
+ (BOOL)_checkElement:(id)element {
  NSError *error;
  BOOL success = [gToolkit checkElement:element error:&error];
  if (error) {
    self.failureHandler(error);
  }
  return success;
}

/**
 Executes the currently installed checks on all the elements of the accessibility tree under the
 given root elements. In case of failures, the failure handler is invoked.

 @param rootElements An array of root elements.
 @return @c NO if any of the checks failed, @c YES otherwise.
 */
+ (BOOL)_checkAllElementsFromRootElements:(NSArray *)rootElements {
  NSError *error;
  BOOL success = [gToolkit checkAllElementsFromRootElements:rootElements error:&error];
  if (error) {
    self.failureHandler(error);
  }
  return success;
}

/**
 Notification handler for handling gtxTestCaseDidBeginNotification.

 @param notification The notification that was posted.
 */
+ (void)_testCaseDidBegin:(NSNotification *)notification {
  // check if new test class has started
  gIsInTearDown = NO;
  Class currentTestClass = notification.userInfo[gtxTestClassUserInfoKey];
  SEL currentTestSelector =
      ((NSInvocation *)notification.userInfo[gtxTestInvocationUserInfoKey]).selector;
  GTXInstallOptions *currentTestCaseOptions = nil;
  for (GTXInstallOptions *options in gIntsallOptions) {
    if ([options.suite hasTestCaseWithClass:currentTestClass
                                 testMethod:currentTestSelector]) {
      currentTestCaseOptions = options;
      break;
    }
  }
  if (gCurrentOptions != currentTestCaseOptions) {
    gCurrentOptions = currentTestCaseOptions;
    if (gCurrentOptions) {
      gToolkit = [[GTXToolKit alloc] init];
      NSAssert(gCurrentOptions.checks && gCurrentOptions.checks.count > 0,
               @"At least one check must be installed!");
      for (id<GTXChecking> check in gCurrentOptions.checks) {
        [gToolkit registerCheck:check];
      }
      for (id<GTXBlacklisting> blacklist in gCurrentOptions.elementBlacklist) {
        [gToolkit registerBlacklist:blacklist];
      }
    }
  }
}

/**
 Notification handler for handling gtxTestCaseDidEndNotification.

 @param notification The notification that was posted.
 */
+ (void)_testCaseDidTearDown:(NSNotification *)notification {
  if (gIsInTearDown) {
    return;
  }
  gIsInTearDown = YES;

  if (gCurrentOptions) {
    // Run all the checks.
    UIWindow *window = [UIApplication sharedApplication].keyWindow;
    if (window) {
      [self _checkAllElementsFromRootElements:@[window]];
    }
  }
}

/**
 Notification handler for handling gtxTestInteractionDidBeginNotification.

 @param notification The notification that was posted.
 */
+ (void)_testInteractionDidBegin:(NSNotification *)notification {
  if (!gIsInInteraction) {
    [self _checkAllElementsFromRootElements:@[[UIApplication sharedApplication].keyWindow.window]];
  }
  gIsInInteraction = YES;
}

/**
 Notification handler for handling gtxTestInteractionDidEndNotification.

 @param notification The notification that was posted.
 */
+ (void)_testInteractionDidFinish:(NSNotification *)notification {
  gIsInInteraction = NO;
}
@end
