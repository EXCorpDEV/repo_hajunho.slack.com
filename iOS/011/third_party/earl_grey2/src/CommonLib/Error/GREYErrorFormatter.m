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

#import "GREYErrorFormatter.h"

#import "GREYFatalAsserts.h"
#import "GREYError+Private.h"
#import "GREYError.h"
#import "GREYErrorConstants.h"
#import "GREYObjectFormatter.h"
#import "NSError+GREYCommon.h"

#pragma mark - UI Hierarchy Keys

static NSString *const kHierarchyHeaderKey = @"UI Hierarchy (Back to front):\n";
static NSString *const kErrorPrefix = @"EarlGrey Encountered an Error:";

#pragma mark - GREYErrorFormatter

@implementation GREYErrorFormatter

#pragma mark - Public Methods

+ (NSString *)formattedDescriptionForError:(GREYError *)error {
  NSMutableString *logger = [[NSMutableString alloc] init];
  NSString *exceptionReason = error.userInfo[kErrorFailureReasonKey];
  if (exceptionReason) {
    [logger appendFormat:@"\n%@", exceptionReason];
  }

  // There shouldn't be a recovery suggestion for a wrappeed error of an underlying error.
  if (!error.nestedError) {
    NSString *recoverySuggestion = error.userInfo[kErrorDetailRecoverySuggestionKey];
    if (recoverySuggestion) {
      [logger appendFormat:@"\n\n%@", recoverySuggestion];
    }
  }

  NSString *elementMatcher = error.userInfo[kErrorDetailElementMatcherKey];
  if (elementMatcher) {
    [logger appendFormat:@"\n\n%@:\n%@", kErrorDetailElementMatcherKey, elementMatcher];
  }

  NSString *failedConstraints = error.userInfo[kErrorDetailConstraintRequirementKey];
  if (failedConstraints) {
    [logger appendFormat:@"\n\n%@:\n%@", kErrorDetailConstraintRequirementKey, failedConstraints];
  }

  NSString *elementDescription = error.userInfo[kErrorDetailElementDescriptionKey];
  if (elementDescription) {
    [logger appendFormat:@"\n\n%@:\n%@", kErrorDetailElementDescriptionKey, elementDescription];
  }

  NSString *assertionCriteria = error.userInfo[kErrorDetailAssertCriteriaKey];
  if (assertionCriteria) {
    [logger appendFormat:@"\n\n%@:\n%@", kErrorDetailAssertCriteriaKey, assertionCriteria];
  }
  NSString *actionCriteria = error.userInfo[kErrorDetailActionNameKey];
  if (actionCriteria) {
    [logger appendFormat:@"\n\n%@:\n%@", kErrorDetailActionNameKey, actionCriteria];
  }

  NSArray<NSString *> *multipleElementsMatched = error.multipleElementsMatched;
  if (multipleElementsMatched) {
    [logger appendFormat:@"\n\n%@:", kErrorDetailElementsMatchedKey];
    [multipleElementsMatched
        enumerateObjectsUsingBlock:^(NSString *element, NSUInteger index, BOOL *stop) {
          // Numbered list of all elements that were matched, starting at 1.
          [logger appendFormat:@"\n\n\t%lu. %@", (unsigned long)index + 1, element];
        }];
  }

  NSString *searchActionInfo = error.userInfo[kErrorDetailSearchActionInfoKey];
  if (searchActionInfo) {
    [logger appendFormat:@"\n\n%@", searchActionInfo];
  }

  NSString *nestedError = error.nestedError.description;
  if (nestedError) {
    [logger appendFormat:@"\n\n*********** Underlying Error ***********:\n%@", nestedError];
  }

  NSString *hierarchy = error.appUIHierarchy;
  if (hierarchy) {
    [logger appendFormat:@"\n\n%@\n%@", kHierarchyHeaderKey, hierarchy];
  }

  return [NSString stringWithFormat:@"%@\n", logger];
}

@end
