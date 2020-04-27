//
//  GlobalSettings.swift
//  swift_settings
//
//  Created by Junho HA on 2018. 9. 27..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

extension String {
    func pwd(_ x: Any)  {
        print("pwd_", String(describing: x.self))
    }
}

struct _logLevel: OptionSet {
    let rawValue: Int
    
    static let critical    = _logLevel(rawValue: 1 << 0)
    static let major  = _logLevel(rawValue: 1 << 1)
    static let minor   = _logLevel(rawValue: 1 << 2)
    static let callBack   = _logLevel(rawValue: 1 << 3)
    static let infiniteLoop = _logLevel(rawValue: 1 << 4)
    static let resourceLeak = _logLevel(rawValue: 1 << 5)
    static let memoryJobs = _logLevel(rawValue: 1 << 6)
    static let library = _logLevel(rawValue: 1 << 7)
    static let all: _logLevel = [.critical, .major, .minor, .callBack, .infiniteLoop, .resourceLeak, .memoryJobs, .library]
}

class GlobalSettings {
    var logLevel : _logLevel
    
    private init() {
        //        logLevel = .all
        //        logLevel = .critical
        logLevel = .minor
    }
    
    static let shared = GlobalSettings()
}
