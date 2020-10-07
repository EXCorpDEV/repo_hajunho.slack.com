//
// Copyright 2017 Google Inc.
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

#import "XCTestCase+GREYTest.h"

#include <objc/runtime.h>

#import "GREYFatalAsserts.h"
#import "GREYTestApplicationDistantObject+Private.h"
#import "GREYFrameworkException.h"
#import "GREYSwizzler.h"
#import "GREYTestCaseInvocation.h"

/**
 * Stack of XCTestCase objects being being executed. This enables the tracking of different nested
 * tests that have been invoked. If empty, then the run is outside the context of a running test.
 */
static NSMutableArray<XCTestCase *> *gExecutingTestCaseStack;

/** Block which will be called when EarlGrey detects that the app-under-test has crashed. */
static void (^gHostApplicationCrashHandler)(void);

/** The port number of the last app-under-test which has crashed. */
static uint16_t gHostApplicationPortForLastCrash;

/**
 * Name of the exception that's thrown to interrupt current test execution.
 */
static NSString *const kInternalTestInterruptException = @"EarlGreyInternalTestInterruptException";

// Extern constants.
NSString *const kGREYXCTestCaseInstanceWillSetUp = @"GREYXCTestCaseInstanceWillSetUp";
NSString *const kGREYXCTestCaseInstanceDidSetUp = @"GREYXCTestCaseInstanceDidSetUp";
NSString *const kGREYXCTestCaseInstanceWillTearDown = @"GREYXCTestCaseInstanceWillTearDown";
NSString *const kGREYXCTestCaseInstanceDidTearDown = @"GREYXCTestCaseInstanceDidTearDown";
NSString *const kGREYXCTestCaseInstanceDidPass = @"GREYXCTestCaseInstanceDidPass";
NSString *const kGREYXCTestCaseInstanceDidFail = @"GREYXCTestCaseInstanceDidFail";
NSString *const kGREYXCTestCaseInstanceDidFinish = @"GREYXCTestCaseInstanceDidFinish";
NSString *const kGREYXCTestCaseNotificationKey = @"GREYXCTestCaseNotificationKey";

/**
 * Checks if there's an app-under-test crash which hasn't been handled yet. If that's the case,
 * @c handler will be invoked. @c handler can indicate that the crash has been handled by returning
 * @c YES. If @c NO is returned by @c handler, the next invocation to this method will again invoke
 * the passed in @c handler to handle the crash.
 *
 * @param handler The block that will be invoked when there is an unhandled app-under-test crash.
 */
static void CheckUnhandledHostApplicationCrashWithHandler(BOOL (^handler)(void));

@implementation XCTestCase (GREYTest)

+ (void)load {
  GREYSwizzler *swizzler = [[GREYSwizzler alloc] init];
  BOOL swizzleSuccess = [swizzler swizzleClass:self
                         replaceInstanceMethod:@selector(invokeTest)
                                    withMethod:@selector(grey_invokeTest)];
  GREYFatalAssertWithMessage(swizzleSuccess, @"Cannot swizzle XCTestCase::invokeTest");
  SEL recordFailureSelector;
  SEL swizzledRecordFailureSelector;
#if defined(__IPHONE_14_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0
  recordFailureSelector = @selector(recordIssue:);
  swizzledRecordFailureSelector = @selector(grey_recordIssue:);
#else
  recordFailureSelector = @selector(recordFailureWithDescription:inFile:atLine:expected:);
  swizzledRecordFailureSelector = @selector(grey_recordFailureWithDescription:
                                                                       inFile:atLine:expected:);
#endif
  swizzleSuccess = [swizzler swizzleClass:self
                    replaceInstanceMethod:recordFailureSelector
                               withMethod:swizzledRecordFailureSelector];
  GREYFatalAssertWithMessage(swizzleSuccess, @"Cannot swizzle XCTestCase::%@",
                             NSStringFromSelector(recordFailureSelector));
  gExecutingTestCaseStack = [[NSMutableArray alloc] init];
}

+ (void)grey_setHostApplicationCrashHandler:(nullable void (^)(void))hostApplicationCrashHandler {
  GREYFatalAssertWithMessage([NSThread isMainThread],
                             @"You must set the crash handler on main thread.");
  CheckUnhandledHostApplicationCrashWithHandler(^{
    NSLog(@"WARNING: The crash handler is overriden right after the crash of app-under-test. This "
          @"may cause the crash being handled in an unexpected way.");
    return NO;
  });
  gHostApplicationCrashHandler = hostApplicationCrashHandler;
}

+ (XCTestCase *)grey_currentTestCase {
  return [gExecutingTestCaseStack lastObject];
}

#if defined(__IPHONE_14_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0
- (void)grey_recordIssue:(XCTIssue *)issue {
  [self grey_setStatus:kGREYXCTestCaseStatusFailed];
  INVOKE_ORIGINAL_IMP1(void, @selector(grey_recordIssue:), issue);
}
#else
- (void)grey_recordFailureWithDescription:(NSString *)description
                                   inFile:(NSString *)filePath
                                   atLine:(NSUInteger)lineNumber
                                 expected:(BOOL)expected {
  [self grey_setStatus:kGREYXCTestCaseStatusFailed];
  INVOKE_ORIGINAL_IMP4(void, @selector(grey_recordFailureWithDescription:inFile:atLine:expected:),
                       description, filePath, lineNumber, expected);
}
#endif

- (NSString *)grey_testMethodName {
  // XCTest.name is represented as "-[<testClassName> <testMethodName>]"
  NSCharacterSet *charsetToStrip =
      [NSMutableCharacterSet characterSetWithCharactersInString:@"-[]"];

  // Resulting string after stripping: <testClassName> <testMethodName>
  NSString *strippedName = [self.name stringByTrimmingCharactersInSet:charsetToStrip];
  // Split string by whitespace.
  NSArray<NSString *> *testClassAndTestMethods =
      [strippedName componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];

  // Test method name will be 2nd item in the array.
  if (testClassAndTestMethods.count <= 1) {
    return nil;
  } else {
    return [testClassAndTestMethods objectAtIndex:1];
  }
}

- (NSString *)grey_testClassName {
  return NSStringFromClass([self class]);
}

- (GREYXCTestCaseStatus)grey_status {
  id status = objc_getAssociatedObject(self, @selector(grey_status));
  return (GREYXCTestCaseStatus)[status unsignedIntegerValue];
}

- (void)grey_markAsFailedAtLine:(NSUInteger)line
                         inFile:(NSString *)file
                    description:(NSString *)description {
  [self grey_recordFailure:file line:line description:description];
  // If the test fails outside of the main thread in a nested runloop, it will not be interrupted
  // until it's back in the outer most runloop. Raise an exception to interrupt the test immediately
  [[GREYFrameworkException exceptionWithName:kInternalTestInterruptException
                                      reason:@"Immediately halt execution of testcase"] raise];
}

#pragma mark - Private

- (BOOL)grey_isSwizzled {
  return [objc_getAssociatedObject([self class], @selector(grey_isSwizzled)) boolValue];
}

- (void)grey_markSwizzled {
  objc_setAssociatedObject([self class], @selector(grey_isSwizzled), @(YES),
                           OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

- (void)grey_invokeTest {
  self.continueAfterFailure = YES;
  @autoreleasepool {
    if (![self grey_isSwizzled]) {
      GREYSwizzler *swizzler = [[GREYSwizzler alloc] init];
      Class selfClass = [self class];
      // Swizzle the setUp and tearDown for this test to allow observing different execution states
      // of the test.
      IMP setUpIMP = [self methodForSelector:@selector(grey_setUp)];
      BOOL swizzleSuccess = [swizzler swizzleClass:selfClass
                                 addInstanceMethod:@selector(grey_setUp)
                                withImplementation:setUpIMP
                      andReplaceWithInstanceMethod:@selector(setUp)];
      GREYFatalAssertWithMessage(swizzleSuccess, @"Cannot swizzle %@ setUp",
                                 NSStringFromClass(selfClass));

      // Swizzle tearDown.
      IMP tearDownIMP = [self methodForSelector:@selector(grey_tearDown)];
      swizzleSuccess = [swizzler swizzleClass:selfClass
                            addInstanceMethod:@selector(grey_tearDown)
                           withImplementation:tearDownIMP
                 andReplaceWithInstanceMethod:@selector(tearDown)];
      GREYFatalAssertWithMessage(swizzleSuccess, @"Cannot swizzle %@ tearDown",
                                 NSStringFromClass(selfClass));
      [self grey_markSwizzled];
    }

    // Change invocation type to GREYTestCaseInvocation to set grey_status to failed if the test
    // method throws an exception. This ensure grey_status is accurate in the test case teardown.
    Class originalInvocationClass =
        object_setClass(self.invocation, [GREYTestCaseInvocation class]);

    @try {
      [gExecutingTestCaseStack addObject:self];
      [self grey_setStatus:kGREYXCTestCaseStatusUnknown];
      INVOKE_ORIGINAL_IMP(void, @selector(grey_invokeTest));

      // The test may have been marked as failed if a failure was recorded with the
      // recordFailureWithDescription:... method. In this case, we can't consider the test has
      // passed.
      if ([self grey_status] != kGREYXCTestCaseStatusFailed) {
        [self grey_setStatus:kGREYXCTestCaseStatusPassed];
      }
    } @catch (NSException *exception) {
      [self grey_setStatus:kGREYXCTestCaseStatusFailed];
      if (![exception.name isEqualToString:kInternalTestInterruptException]) {
        @throw;  // NOLINT
      }
    } @finally {
      switch ([self grey_status]) {
        case kGREYXCTestCaseStatusFailed:
          [self grey_sendNotification:kGREYXCTestCaseInstanceDidFail];
          break;
        case kGREYXCTestCaseStatusPassed:
          [self grey_sendNotification:kGREYXCTestCaseInstanceDidPass];
          break;
        case kGREYXCTestCaseStatusUnknown:
          self.continueAfterFailure = YES;
          [self grey_recordFailure:@__FILE__
                              line:__LINE__
                       description:@"Test has finished with unknown status."];
          break;
      }
      object_setClass(self.invocation, originalInvocationClass);
      [self grey_sendNotification:kGREYXCTestCaseInstanceDidFinish];
      // We only reset the current test case after all possible notifications have been sent.
      [gExecutingTestCaseStack removeLastObject];
    }
  }
}

/**
 * A swizzled implementation for XCTestCase::setUp.
 *
 * @remark These methods need to be added to each instance of XCTestCase because we don't expect
 *         test to invoke <tt> [super setUp] </tt>.
 */
- (void)grey_setUp {
  [self grey_sendNotification:kGREYXCTestCaseInstanceWillSetUp];
  CheckUnhandledHostApplicationCrashWithHandler(^{
    if (gHostApplicationCrashHandler) {
      gHostApplicationCrashHandler();
    }
    return YES;
  });
  INVOKE_ORIGINAL_IMP(void, @selector(grey_setUp));
  [self grey_sendNotification:kGREYXCTestCaseInstanceDidSetUp];
}

/**
 * A swizzled implementation for XCTestCase::tearDown.
 *
 * @remark These methods need to be added to each instance of XCTestCase because we don't expect
 *         tests to invoke <tt> [super tearDown] </tt>.
 */
- (void)grey_tearDown {
  [self grey_sendNotification:kGREYXCTestCaseInstanceWillTearDown];
  CheckUnhandledHostApplicationCrashWithHandler(^{
    if (gHostApplicationCrashHandler) {
      gHostApplicationCrashHandler();
    }
    return YES;
  });
  INVOKE_ORIGINAL_IMP(void, @selector(grey_tearDown));
  [self grey_sendNotification:kGREYXCTestCaseInstanceDidTearDown];
}

/**
 * Posts a notification with the specified @c notificationName using the default
 * NSNotificationCenter and with the @c userInfo containing the current test case.
 *
 * @param notificationName Name of the notification to be posted.
 */
- (void)grey_sendNotification:(NSString *)notificationName {
  NSDictionary<NSString *, id> *userInfo = @{kGREYXCTestCaseNotificationKey : self};
  [[NSNotificationCenter defaultCenter] postNotificationName:notificationName
                                                      object:self
                                                    userInfo:userInfo];
}

#pragma mark - Package Internal

- (void)grey_setStatus:(GREYXCTestCaseStatus)status {
  objc_setAssociatedObject(self, @selector(grey_status), @(status),
                           OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

/**
 * Calls the XCTest record methods for recording a failure in the execution of a test.
 *
 * @param filePath    Name of the file which contains the failure.
 * @param line        Line number in the @c filePath where the failure occurred.
 * @param description Full description of the failure. Utilized as compactDescription in iOS 14+.
 */
- (void)grey_recordFailure:(NSString *)filePath
                      line:(NSUInteger)line
               description:(NSString *)description {
#if defined(__IPHONE_14_0)
  XCTSourceCodeLocation *location =
      [[XCTSourceCodeLocation alloc] initWithFilePath:filePath lineNumber:(NSInteger)line];
  XCTSourceCodeContext *context = [[XCTSourceCodeContext alloc] initWithLocation:location];
  XCTIssue *issue = [[XCTIssue alloc] initWithType:XCTIssueTypeUncaughtException
                                compactDescription:description
                               detailedDescription:nil
                                 sourceCodeContext:context
                                   associatedError:nil
                                       attachments:@[]];
  [self recordIssue:issue];
#else
  [self recordFailureWithDescription:description inFile:filePath atLine:line expected:NO];
#endif
}

@end

static void CheckUnhandledHostApplicationCrashWithHandler(BOOL (^handler)(void)) {
  GREYFatalAssertWithMessage([NSThread isMainThread],
                             @"Application crash should be checked on main thread.");
  GREYTestApplicationDistantObject *testDistantObject =
      GREYTestApplicationDistantObject.sharedInstance;
  if (testDistantObject.hostApplicationTerminated) {
    // testDistantObject.hostPort won't be 0 if testDistantObject.hostApplicationTerminated is true.
    uint16_t currentHostPort = testDistantObject.hostPort;
    if (currentHostPort != gHostApplicationPortForLastCrash) {
      if (handler()) {
        gHostApplicationPortForLastCrash = currentHostPort;
      }
    }
  }
}
