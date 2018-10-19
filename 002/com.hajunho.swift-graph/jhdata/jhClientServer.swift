//
//  jhClientServer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 18..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhClientServer {
    
    private var listeners = [observer_p]()
    
    public func notiDataDowloadFinish() {
        for x in listeners {
            x.jhRedraw()
        }
    }
    
    //    public func getData() -> Array<CGFloat> {
    //        return self.mValuesOfDatas
    //    }
    //
    //    public func setData(x: Array<CGFloat>) {
    //        self.mValuesOfDatas = x
    //    }
}
